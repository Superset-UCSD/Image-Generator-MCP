from __future__ import annotations

from pathlib import Path

from PIL import Image

from image_gen_mcp.config import AppConfig
from image_gen_mcp.mcp_tools import ImageGenService
from image_gen_mcp.models import LocalEditArgs


def test_local_edit_resize_crop_and_manifest(tmp_path: Path) -> None:
    cfg = AppConfig(out_dir=str(tmp_path / "assets_out"))
    service = ImageGenService(cfg)

    source_id = "aaaaaaaaaaaaaaaa"
    src_img = Image.new("RGBA", (100, 80), "#ff0000ff")
    service.store.save_image(source_id, src_img)

    resize_result = service.local_edit(
        LocalEditArgs(
            input_asset_id=source_id,
            op="resize",
            params={"width": 50, "height": 50, "mode": "contain", "background": "#00000000"},
            tags=["test"],
        )
    )
    resized = Image.open(resize_result["path"])
    assert resized.size == (50, 50)

    crop_result = service.local_edit(
        LocalEditArgs(
            input_asset_id=source_id,
            op="crop",
            params={"x": 10, "y": 10, "w": 30, "h": 20},
            tags=["test"],
        )
    )
    cropped = Image.open(crop_result["path"])
    assert cropped.size == (30, 20)

    manifest = service.store.manifest_tail(100)
    assert "\"event\": \"asset\"" in manifest
    assert resize_result["asset_id"] in manifest
    assert crop_result["asset_id"] in manifest
