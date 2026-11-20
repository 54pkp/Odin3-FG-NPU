"""Inspect or stage just the benchmark files from the official QAIRT ZIP."""
from __future__ import annotations

import argparse
import io
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PACKAGE = ("https://softwarecenter.qualcomm.com/api/download/software/sdks/"
           "Qualcomm_AI_Runtime_Community/All/2.49.0.260730/v2.49.0.260730.zip")
LINUX = "aarch64-oe-linux-gcc11.2"
TOOLS = {"qnn-context-binary-generator", "qnn-net-run", "qnn-profile-viewer"}
LIBS = {"libQnnHtp.so", "libQnnHtpPrepare.so", "libQnnHtpV79Stub.so",
        "libQnnHtpNetRunExtensions.so", "libQnnModelDlc.so", "libQnnSystem.so"}


class RangeReader(io.RawIOBase):
    def __init__(self, url: str) -> None:
        self._session = requests.Session()
        self._session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])))
        with self._session.get(url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
                               stream=True, timeout=(10, 20)) as result:
            result.raise_for_status()
            match = re.fullmatch(r"bytes 0-0/(\d+)", result.headers.get("Content-Range", ""))
            if result.status_code != 206 or not match:
                raise OSError("QAIRT server did not honor the size probe")
            self._size = int(match.group(1))
            self._url = result.url
        self._offset = 0
        self._cache_at = -1
        self._cache = b""

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._offset

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        position = {io.SEEK_SET: 0, io.SEEK_CUR: self._offset, io.SEEK_END: self._size}[whence] + offset
        if position < 0:
            raise ValueError("Negative ZIP offset")
        self._offset = position
        return position

    def read(self, size: int = -1) -> bytes:
        remaining = max(0, self._size - self._offset)
        count = remaining if size < 0 else min(size, remaining)
        if not count:
            return b""
        here = self._offset
        delta = here - self._cache_at
        if not (0 <= delta and delta + count <= len(self._cache)):
            end = min(self._size, here + max(count, 262_144)) - 1
            headers = {"Range": f"bytes={here}-{end}", "Accept-Encoding": "identity"}
            with self._session.get(self._url, headers=headers, stream=True, timeout=(10, 30)) as result:
                result.raise_for_status()
                if result.status_code != 206 or result.headers.get("Content-Range") != f"bytes {here}-{end}/{self._size}":
                    raise OSError(f"QAIRT server returned an unexpected range for {here}-{end}")
                self._cache = result.content
                self._cache_at = here
            if len(self._cache) != end - here + 1:
                raise OSError("Truncated range response")
            delta = 0
        self._offset += count
        return self._cache[delta:delta + count]

    def close(self) -> None:
        self._session.close()
        super().close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with RangeReader(PACKAGE) as source, zipfile.ZipFile(source) as archive:
        found = set()
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            bits = path.parts
            if len(bits) < 4:
                continue
            if bits[-3:-1] == ("bin", LINUX) and path.name in TOOLS:
                relative = Path("bin", path.name)
            elif bits[-3:-1] == ("lib", LINUX) and path.name in LIBS:
                relative = Path("lib", path.name)
            elif bits[-3:-1] == ("hexagon-v79", "unsigned") and re.fullmatch(
                r"(libQairtHtp|libQnnHtp)V79(?:Skel)?\.so", path.name
            ):
                relative = Path("hexagon", path.name)
            else:
                continue
            if relative in found:
                raise ValueError(f"Duplicate member for {relative}")
            found.add(relative)
            print(f"{relative} {entry.file_size:,} bytes", flush=True)
            if args.output is not None:
                output = args.output / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                if not output.is_file():
                    temporary = output.with_suffix(output.suffix + ".part")
                    with archive.open(entry) as input_file, temporary.open("wb") as destination:
                        shutil.copyfileobj(input_file, destination)
                    temporary.replace(output)
                if output.stat().st_size != entry.file_size:
                    raise OSError(f"Unexpected size for {output}")
        missing = {*[Path("bin", name) for name in TOOLS], *[Path("lib", name) for name in LIBS]} - found
        if missing:
            raise FileNotFoundError(", ".join(map(str, sorted(missing))))


if __name__ == "__main__":
    main()
