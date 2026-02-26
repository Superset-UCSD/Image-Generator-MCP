from __future__ import annotations

import os
import shutil
import tomllib
from pathlib import Path
from typing import Any

import tomli_w


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))


def codex_config_path() -> Path:
    return codex_home() / "config.toml"


def _default_uv_path() -> str:
    return shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")


def setup_codex_server(repo_dir: Path, *, server_name: str = "image-gen-mcp") -> Path:
    cfg_path = codex_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {}
    if cfg_path.exists():
        data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))

    mcp_servers = data.get("mcp_servers")
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
        data["mcp_servers"] = mcp_servers

    mcp_servers[server_name] = {
        "command": _default_uv_path(),
        "args": ["run", "image-gen-mcp"],
        "cwd": str(repo_dir.resolve()),
    }

    cfg_path.write_text(tomli_w.dumps(data), encoding="utf-8")
    return cfg_path
