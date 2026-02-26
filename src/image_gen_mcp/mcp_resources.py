from __future__ import annotations

from typing import Any

from .cache import validate_asset_id
from .mcp_tools import ImageGenService


def register_resources(mcp: Any, service: ImageGenService) -> None:
    @mcp.resource("asset://image/{asset_id}")
    def image_resource(asset_id: str) -> bytes:
        aid = validate_asset_id(asset_id)
        path = service.store.image_path(aid)
        return path.read_bytes()

    @mcp.resource("asset://meta/{asset_id}")
    def meta_resource(asset_id: str) -> str:
        aid = validate_asset_id(asset_id)
        path = service.store.meta_path(aid)
        return path.read_text(encoding="utf-8")

    @mcp.resource("asset://manifest/latest{?n}")
    def manifest_latest(n: int = 100) -> str:
        count = min(max(int(n), 1), 5000)
        return service.store.manifest_tail(count)

    @mcp.resource("asset://manifest/latest/{n}")
    def manifest_latest_path(n: str) -> str:
        count = min(max(int(n), 1), 5000)
        return service.store.manifest_tail(count)
