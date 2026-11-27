"""Verify and manage versioned Odin 3 FG runtime archives on ArmadaOS."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import file_digest
import json
import os
from pathlib import Path
import re
from shutil import copyfileobj
import stat
import subprocess
import sys
import tarfile
from tempfile import NamedTemporaryFile, TemporaryDirectory


EXECUTABLES = {"odin3-fg-worker", "odin3-fg.sh", "fastrpc/bin/cdsprpcd"}
BINARIES = {
    "odin3-fg-worker",
    "fastrpc/bin/cdsprpcd",
    "lib/libmako-render.so",
    "lib/libodin3-wsi-dma-capture.so",
    "fastrpc/lib/libcdsprpc.so",
    "fastrpc/lib/libcdsp_default_listener.so.1",
}
MANIFEST = "runtime-manifest.json"
LAYER_MANIFEST = "share/vulkan/implicit_layer.d/VkLayer_MAKO_render.json"
WSI_CAPTURE = "lib/libodin3-wsi-dma-capture.so"
SHARED_INPUT_PROTOCOL = "0x4f33000b"
REQUIRED_FILES = EXECUTABLES | BINARIES | {
    "profiles.py",
    "fastrpc/odin3-fg-cdsp.service",
    "fastrpc/licenses/FastRPC.txt",
    "fastrpc/licenses/LibYAML.txt",
    LAYER_MANIFEST,
}
DIGEST = re.compile(r"[0-9a-f]{64}")
SELECTION = "active.json"


def validate_manifest(receipt):
    if (not isinstance(receipt, dict) or type(receipt.get("schema_version")) is not int
            or receipt["schema_version"] != 1 or receipt.get("architecture") != "aarch64"):
        raise ValueError("Unsupported runtime manifest")
    files = receipt.get("files")
    shared = receipt.get("shared_input")
    required = REQUIRED_FILES
    if shared is not None:
        if (not isinstance(shared, dict) or shared.get("protocol") != SHARED_INPUT_PROTOCOL
                or not isinstance(shared.get("source"), dict)):
            raise ValueError("Unsupported shared-input build receipt")
    if not isinstance(files, dict) or files.keys() != required:
        raise ValueError("Runtime manifest does not match the package contract")
    for name, expected in files.items():
        if (not isinstance(expected, dict) or type(expected.get("size")) is not int
                or expected["size"] < 0 or not isinstance(expected.get("sha256"), str)
                or not DIGEST.fullmatch(expected["sha256"])):
            raise ValueError(f"Invalid runtime manifest entry: {name}")
    return files


def verify_file(name, stream, size, expected):
    if size != expected["size"]:
        raise ValueError(f"Runtime size mismatch: {name}")
    if name in BINARIES:
        header = stream.read(20)
        if header[:6] != b"\x7fELF\x02\x01" or header[18:20] != b"\xb7\x00":
            raise ValueError(f"Expected a little-endian ARM64 ELF: {name}")
        stream.seek(0)
    if file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
        raise ValueError(f"Runtime hash mismatch: {name}")


def verify_archive(path, destination=None):
    expected_entries = REQUIRED_FILES | {MANIFEST}
    with tarfile.open(path, "r:") as archive:
        entries = {}
        for member in archive:
            name = member.name
            if not member.isreg() or name not in expected_entries or name in entries:
                raise ValueError(f"Unexpected or duplicate archive entry: {name!r}")
            if member.mode != (0o755 if name in EXECUTABLES else 0o644):
                raise ValueError(f"Unexpected permissions for archive entry: {name}")
            entries[name] = member
        if entries.keys() != expected_entries:
            raise ValueError("Runtime archive is missing required entries")
        manifest = entries.pop(MANIFEST)
        with archive.extractfile(manifest) as stream:
            receipt = json.load(stream)
        files = validate_manifest(receipt)
        if entries.keys() != files.keys():
            raise ValueError("Runtime archive does not match its manifest")
        for name, member in entries.items():
            with archive.extractfile(member) as stream:
                verify_file(name, stream, member.size, files[name])
                if destination is not None:
                    target = destination / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stream.seek(0)
                    with target.open("xb") as output:
                        copyfileobj(stream, output)
                    target.chmod(member.mode)
        if destination is not None:
            with archive.extractfile(manifest) as stream, (destination / MANIFEST).open("xb") as out:
                copyfileobj(stream, out)
            (destination / MANIFEST).chmod(0o644)
        return receipt


def read_regular(root, relative):
    path = root / relative
    if path.resolve() != path or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Unexpected runtime file type: {relative}")
    return path


def verify_release(path):
    if path.resolve() != path or not path.is_dir():
        raise ValueError(f"Runtime release unavailable: {path.name}")
    actual = {entry.relative_to(path).as_posix() for entry in path.rglob("*")
              if entry.is_symlink() or not entry.is_dir()}
    receipt = json.loads(read_regular(path, MANIFEST).read_text(encoding="utf-8"))
    files = validate_manifest(receipt)
    if actual != files.keys() | {MANIFEST}:
        raise ValueError(f"Installed runtime does not match the package contract: {path.name}")
    for name, entry in files.items():
        file = read_regular(path, name)
        info = file.stat()
        if stat.S_IMODE(info.st_mode) != (0o755 if name in EXECUTABLES else 0o644):
            raise ValueError(f"Unexpected permissions for installed runtime file: {name}")
        with file.open("rb") as stream:
            verify_file(name, stream, info.st_size, entry)
    return receipt


def digest_argument(text):
    value = text.lower()
    if not DIGEST.fullmatch(value):
        raise argparse.ArgumentTypeError("Expected a SHA-256 digest")
    return value


@contextmanager
def release_lock(root):
    import fcntl

    root.mkdir(mode=0o755, parents=True, exist_ok=True)
    fd = os.open(root / ".install.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def stage(root, archive, digest):
    versions = root / "versions"
    versions.mkdir(mode=0o755, exist_ok=True)
    if versions.is_symlink():
        raise ValueError("The runtime versions directory must be local to the install root")
    release = versions / digest
    with TemporaryDirectory(prefix=".stage-", dir=root) as temporary:
        work = Path(temporary)
        snapshot = work / "bundle.tar"
        with archive.open("rb") as source, snapshot.open("xb") as output:
            copyfileobj(source, output)
        with snapshot.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("Runtime archive SHA-256 does not match the expected digest")
        if release.exists() or release.is_symlink():
            if verify_release(release) != verify_archive(snapshot):
                raise ValueError("Installed release differs from the verified package")
        else:
            payload = work / "payload"
            payload.mkdir(mode=0o755)
            verify_archive(snapshot, payload)
            payload.rename(release)
    return release


def selection(root):
    try:
        state = json.loads(read_regular(root, SELECTION).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"active": None, "previous": None}
    if (not isinstance(state, dict) or state.keys() != {"active", "previous"}
            or not isinstance(state["active"], str) or not DIGEST.fullmatch(state["active"])
            or (state["previous"] is not None
                and (not isinstance(state["previous"], str) or not DIGEST.fullmatch(state["previous"])
                     or state["active"] == state["previous"]))):
        raise ValueError("Invalid runtime selection")
    return state


def publish_selection(root, active, previous):
    with NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".select-", dir=root, delete=False) as stream:
        pending = Path(stream.name)
        try:
            json.dump({"active": active, "previous": previous}, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            pending.chmod(0o644)
            pending.replace(root / SELECTION)
        finally:
            pending.unlink(missing_ok=True)


def launch(root, command):
    try:
        active = selection(root)["active"]
        if active is not None:
            release = root / "versions" / active
            launcher = read_regular(release, "odin3-fg.sh")
            os.execv(launcher, [str(launcher), "--", *command])
        reason = "No active runtime"
    except (OSError, ValueError) as error:
        reason = str(error)
    print(f"Odin 3 FG: {reason}. Starting the original command.", file=sys.stderr)
    os.execvp(command[0], command)


def main():
    default_data = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_data / "odin3-fg")
    commands = parser.add_subparsers(dest="action", required=True)
    add = commands.add_parser("stage")
    add.add_argument("archive", type=Path)
    add.add_argument("--sha256", required=True, type=digest_argument)
    activate = commands.add_parser("activate")
    activate.add_argument("digest", type=digest_argument)
    activate.add_argument("health_command", nargs=argparse.REMAINDER,
                          help="Command receiving the absolute candidate release path as its last argument")
    commands.add_parser("rollback")
    commands.add_parser("status")
    commands.add_parser("launch").add_argument("game_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    try:
        if args.action == "launch":
            command = args.game_command
            if command[:1] == ["--"]:
                command = command[1:]
            if not command:
                parser.error("launch requires the original game command")
            launch(root, command)
        elif args.action == "status":
            print(json.dumps(selection(root)))
        else:
            with release_lock(root):
                if args.action == "stage":
                    print(stage(root, args.archive, args.sha256))
                    return
                state = selection(root)
                if args.action == "rollback":
                    if state["previous"] is None:
                        raise ValueError("No previous runtime to restore")
                    active = state["previous"]
                    verify_release(root / "versions" / active)
                else:
                    active = args.digest
                    command = args.health_command
                    if command[:1] == ["--"]:
                        command = command[1:]
                    if command:
                        release = root / "versions" / active
                        verify_release(release)
                        subprocess.run([*command, str(release)], stdout=sys.stderr, check=True)
                    verify_release(root / "versions" / active)
                if active != state["active"]:
                    publish_selection(root, active, state["active"])
                print(json.dumps(selection(root)))
    except (OSError, ValueError, tarfile.TarError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Odin 3 FG installer: {error}\n")


if __name__ == "__main__":
    main()
