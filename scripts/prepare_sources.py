"""Check out the pinned dependencies and apply the MAKO patch without resetting any worktree."""
import json
from pathlib import Path
import subprocess

from build_inputs import ROOT, SOURCES


def git(directory, *args, **options):
    return subprocess.run(["git", *map(str, args)], cwd=directory, text=True, check=True, **options)


def main():
    spec = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    SOURCES.mkdir(parents=True, exist_ok=True)
    for name, source in spec.items():
        target = SOURCES / name
        revision = source["revision"]
        if not target.exists():
            git(SOURCES, "clone", "--filter=blob:none", "--no-checkout", source["url"], target)
            git(target, "checkout", "--detach", revision)
        checkout = git(target, "rev-parse", "--show-toplevel", capture_output=True).stdout.strip()
        head = git(target, "rev-parse", "HEAD", capture_output=True).stdout.strip()
        if Path(checkout).resolve() != target.resolve() or head != revision:
            raise RuntimeError(f"{target} must be a separate checkout at {revision}")
        origin = git(target, "remote", "get-url", "origin", capture_output=True).stdout.strip()
        if origin.removesuffix(".git") != source["url"].removesuffix(".git"):
            raise RuntimeError(f"Unexpected origin in {target}")
    mako = SOURCES / "MAKO"
    patch = ROOT / "patches/mako-odin3-anvil.patch"
    status = git(mako, "status", "--porcelain", capture_output=True).stdout
    if status:
        # Only inspect an existing patched input, without overwriting local work.
        git(mako, "apply", "--reverse", "--check", patch)
        print("MAKO patch already present. Existing source changes were preserved.")
    else:
        git(mako, "apply", "--check", patch)
        git(mako, "apply", patch)
    print(f"Pinned sources ready under {SOURCES}")


if __name__ == "__main__":
    main()
