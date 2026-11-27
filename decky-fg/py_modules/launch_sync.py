"""Token-preserving Steam Launch Options editor for Odin 3 FG NPU."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

CMD = "%command%"
LAUNCHER_REL = ".local/share/odin3-fg/bin/odin3-fg-launch"
LAUNCHER_TOKEN = "~/" + LAUNCHER_REL
_RX_TOKEN = re.compile(r'''(?:"[^"]*"?|'[^']*'?|[^\s"'])+''')


def tokens(options: str | None) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _RX_TOKEN.finditer(options or "")]


def words(options: str | None) -> list[str]:
    return [t for t, _, _ in tokens(options)]


def has_cmd(options: str | None) -> bool:
    return CMD in (options or "")


def wrapped_cmd(options: str | None) -> bool:
    return has_cmd(options) and CMD not in words(options)


def is_our_token(tok: str) -> bool:
    cleaned = tok.strip('"\'')
    return cleaned == LAUNCHER_TOKEN or cleaned.endswith("/" + LAUNCHER_REL)


def has_launcher(options: str | None) -> bool:
    return any(is_our_token(t) for t in words(options))


def _cut(s: str, start: int, end: int, prefer_after: bool) -> str:
    after = end < len(s) and s[end].isspace()
    before = start > 0 and s[start - 1].isspace()
    if (prefer_after and after) or (not before and after):
        return s[:start] + s[end + 1 :]
    if before:
        return s[: start - 1] + s[end:]
    return s[:start] + s[end:]


def add_launcher(options: str | None) -> tuple[str, dict, str | None]:
    s = options if options is not None else ""
    if wrapped_cmd(s):
        return s, {"added_launcher": False, "added_cmd": False}, "检测到复合脚本启动项，请手动在 %command% 前加入 ~/.local/share/odin3-fg/bin/odin3-fg-launch"
    if has_launcher(s):
        return s, {"added_launcher": False, "added_cmd": False}, None

    added_cmd = False
    if not has_cmd(s):
        s = CMD + (" " + s if s.strip() else "")
        added_cmd = True

    toks = tokens(s)
    cmd_idx = next((i for i, (t, _, _) in enumerate(toks) if t == CMD), None)
    if cmd_idx is None:
        return s, {"added_launcher": False, "added_cmd": False}, "未找到独立的 %command% 标记"

    _, cmd_start, _ = toks[cmd_idx]
    updated = s[:cmd_start] + LAUNCHER_TOKEN + " " + s[cmd_start:]
    return updated, {"added_launcher": True, "added_cmd": added_cmd}, None


def remove_launcher(options: str | None, state: dict | None = None) -> str | None:
    if options is None:
        return None
    st = state or {}
    if st.get("enabled") and isinstance(st.get("launch_set"), str) and options == st.get("launch_set"):
        if isinstance(st.get("launch_prev"), str):
            return st["launch_prev"]

    s = options
    while True:
        toks = tokens(s)
        found = next(((a, b) for t, a, b in toks if is_our_token(t)), None)
        if not found:
            break
        s = _cut(s, found[0], found[1], True)

    added_cmd = bool((st.get("rec") or {}).get("added_cmd"))
    toks = tokens(s)
    if len(toks) == 1 and toks[0][0] == CMD and (added_cmd or st.get("launch_prev", "") == ""):
        s = ""
    elif added_cmd and toks and toks[0][0] == CMD and [t for t, _, _ in toks].count(CMD) == 1:
        s = _cut(s, toks[0][1], toks[0][2], True).strip()
    return s


def state_path(home: Path, app_id: int) -> Path:
    return home / ".config/odin3-fg/launch-state" / f"{app_id}.json"


def sync_launch_options(home: Path, uid: int, gid: int, app_id: int, enable: bool, current_options: str | None) -> tuple[str | None, str | None]:
    if current_options is None:
        return None, None
    sp = state_path(home, app_id)
    prev_state = None
    try:
        if sp.is_file():
            prev_state = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        prev_state = None

    if enable:
        if has_launcher(current_options):
            return current_options, None
        updated, rec, warn = add_launcher(current_options)
        if warn is None:
            sp.parent.mkdir(parents=True, exist_ok=True)
            os.chown(sp.parent.parent, uid, gid)
            os.chown(sp.parent, uid, gid)
            payload = {
                "enabled": True,
                "launch_prev": current_options,
                "launch_set": updated,
                "rec": rec,
            }
            tmp = sp.with_name(f".{app_id}.tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.chown(tmp, uid, gid)
            tmp.replace(sp)
        return updated, warn
    else:
        restored = remove_launcher(current_options, prev_state)
        try:
            sp.unlink(missing_ok=True)
        except OSError:
            pass
        return restored, None
