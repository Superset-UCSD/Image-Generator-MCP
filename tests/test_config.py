from __future__ import annotations

from pathlib import Path

from image_gen_mcp import config


def test_config_read_write(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)

    path = config.set_token("hf_1234567890")
    assert path == tmp_path / "config.toml"

    cfg = config.load_config()
    assert cfg.hf_token == "hf_1234567890"

    updated = config.set_defaults(steps=42, guidance=6.5, out_dir="./x")
    assert updated.defaults.steps == 42
    assert updated.defaults.guidance == 6.5
    assert updated.out_dir == "./x"


def test_redact_token() -> None:
    assert config.redact_token(None) is None
    assert config.redact_token("abcd") == "****"
    assert config.redact_token("hf_1234567890") == "hf_1...7890"
