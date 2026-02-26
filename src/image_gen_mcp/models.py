from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolAssetResult(BaseModel):
    asset_id: str
    path: str
    mime_type: str = "image/png"
    width: int
    height: int
    model: str | None = None
    prompt_used: str | None = None
    negative_prompt_used: str | None = None
    cache_hit: bool = False
    resource_uri: str
    meta_uri: str
    input_image_hash: str | None = None


class GenerateArgs(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    model: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = None
    force: bool = False
    tags: list[str] | None = None


class AiEditArgs(BaseModel):
    input_asset_id: str | None = None
    input_path: str | None = None
    prompt: str
    negative_prompt: str | None = None
    model: str | None = None
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = None
    strength: float | None = Field(default=0.6, ge=0.0, le=1.0)
    force: bool = False
    tags: list[str] | None = None


class LocalEditArgs(BaseModel):
    input_asset_id: str | None = None
    input_path: str | None = None
    op: str
    params: dict[str, Any] = Field(default_factory=dict)
    force: bool = False
    tags: list[str] | None = None


class BatchItem(BaseModel):
    tool: Literal["image.generate", "image.ai_edit", "image.local_edit"]
    args: dict[str, Any]


class BatchArgs(BaseModel):
    items: list[BatchItem]
    max_parallel: int = Field(default=2, ge=1)


class ListArgs(BaseModel):
    limit: int = Field(default=50, ge=1, le=1000)
    tag: str | None = None
    task: str | None = None


class GetMetaArgs(BaseModel):
    asset_id: str


class HealthcheckArgs(BaseModel):
    smoke: bool = False
