from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from platformdirs import user_config_dir
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


class DefaultsConfig(BaseModel):
    t2i_model: str = DEFAULT_MODEL
    i2i_model: str = DEFAULT_MODEL
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance: float = 7.0
    negative_prompt: str = ""
    max_concurrency: int = 2
    retry_max: int = 6
    retry_base_delay: float = 0.8


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hf_token: str | None = None
    out_dir: str = "./assets_out"
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)


def config_dir() -> Path:
    return Path(user_config_dir("image-gen-mcp"))


def config_path() -> Path:
    return config_dir() / "config.toml"


def ensure_config_dir() -> Path:
    path = config_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config() -> AppConfig:
    path = config_path()
    if not path.exists():
        return AppConfig()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(data)


def save_config(cfg: AppConfig) -> Path:
    ensure_config_dir()
    path = config_path()
    payload = cfg.model_dump(mode="json")
    path.write_text(tomli_w.dumps(payload), encoding="utf-8")
    return path


def set_token(token: str) -> Path:
    cfg = load_config()
    cfg.hf_token = token.strip()
    return save_config(cfg)


def set_defaults(**kwargs: Any) -> AppConfig:
    cfg = load_config()
    for key, value in kwargs.items():
        if value is not None:
            if key == "out_dir":
                continue
            setattr(cfg.defaults, key, value)
    if kwargs.get("out_dir") is not None:
        cfg.out_dir = str(kwargs["out_dir"])
    save_config(cfg)
    return cfg


def resolve_token(api_key_override: str | None = None) -> str | None:
    if api_key_override:
        return api_key_override
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token
    return load_config().hf_token


def effective_config(api_key_override: str | None = None) -> AppConfig:
    cfg = load_config()
    cfg.hf_token = resolve_token(api_key_override)
    env_out_dir = os.getenv("IMAGE_GEN_OUT_DIR")
    if env_out_dir:
        cfg.out_dir = env_out_dir
    return cfg


def redact_token(token: str | None) -> str | None:
    if not token:
        return None
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}...{token[-4:]}"


def shown_config(cfg: AppConfig) -> dict[str, Any]:
    payload = cfg.model_dump(mode="json")
    payload["hf_token"] = redact_token(cfg.hf_token)
    return payload
