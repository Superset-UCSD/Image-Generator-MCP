from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


@dataclass
class LocalEditOutput:
    image: Image.Image | None
    extra: dict[str, Any]


def parse_color(value: str | None) -> str | tuple[int, int, int, int]:
    return value or "#00000000"


def resize_image(image: Image.Image, params: dict[str, Any]) -> Image.Image:
    width = int(params["width"])
    height = int(params["height"])
    mode = params.get("mode", "contain")
    bg = parse_color(params.get("background"))

    if mode == "stretch":
        return image.resize((width, height), Image.Resampling.LANCZOS)

    if mode == "contain":
        src = image.copy()
        src.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (width, height), bg)
        x = (width - src.width) // 2
        y = (height - src.height) // 2
        canvas.paste(src, (x, y))
        return canvas

    if mode == "cover":
        return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)

    raise ValueError("resize mode must be one of contain|cover|stretch")


def crop_image(image: Image.Image, params: dict[str, Any]) -> Image.Image:
    x = int(params["x"])
    y = int(params["y"])
    w = int(params["w"])
    h = int(params["h"])
    return image.crop((x, y, x + w, y + h))


def pad_image(image: Image.Image, params: dict[str, Any]) -> Image.Image:
    left = int(params.get("left", 0))
    right = int(params.get("right", 0))
    top = int(params.get("top", 0))
    bottom = int(params.get("bottom", 0))
    color = parse_color(params.get("color"))
    w = image.width + left + right
    h = image.height + top + bottom
    out = Image.new("RGBA", (w, h), color)
    out.paste(image, (left, top))
    return out


def rotate_image(image: Image.Image, params: dict[str, Any]) -> Image.Image:
    degrees = float(params.get("degrees", 0.0))
    expand = bool(params.get("expand", True))
    fill = parse_color(params.get("bg"))
    return image.rotate(degrees, expand=expand, fillcolor=fill)


def flip_image(image: Image.Image, params: dict[str, Any]) -> Image.Image:
    direction = params.get("direction", "horizontal")
    if direction == "horizontal":
        return ImageOps.mirror(image)
    if direction == "vertical":
        return ImageOps.flip(image)
    raise ValueError("flip direction must be horizontal or vertical")


def convert_image(image: Image.Image, params: dict[str, Any]) -> Image.Image:
    mode = params.get("mode", "RGBA")
    return image.convert(mode)


def quantize_image(image: Image.Image, params: dict[str, Any]) -> Image.Image:
    colors = int(params.get("colors", 64))
    return image.convert("RGBA").quantize(colors=colors).convert("RGBA")


def slice_grid(image: Image.Image, params: dict[str, Any]) -> list[Image.Image]:
    rows = int(params["rows"])
    cols = int(params["cols"])
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")

    cell_w = image.width // cols
    cell_h = image.height // rows
    slices: list[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            x0 = c * cell_w
            y0 = r * cell_h
            x1 = image.width if c == cols - 1 else (c + 1) * cell_w
            y1 = image.height if r == rows - 1 else (r + 1) * cell_h
            slices.append(image.crop((x0, y0, x1, y1)))
    return slices


def nine_slice_meta(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "nine_slice": {
            "left": int(params["l"]),
            "right": int(params["r"]),
            "top": int(params["t"]),
            "bottom": int(params["b"]),
        }
    }


def run_local_edit(
    image: Image.Image,
    op: str,
    params: dict[str, Any],
    *,
    _out_dir: Path,
) -> LocalEditOutput:
    if op == "resize":
        return LocalEditOutput(image=resize_image(image, params), extra={})
    if op == "crop":
        return LocalEditOutput(image=crop_image(image, params), extra={})
    if op == "pad":
        return LocalEditOutput(image=pad_image(image, params), extra={})
    if op == "rotate":
        return LocalEditOutput(image=rotate_image(image, params), extra={})
    if op == "flip":
        return LocalEditOutput(image=flip_image(image, params), extra={})
    if op == "convert":
        return LocalEditOutput(image=convert_image(image, params), extra={})
    if op == "quantize":
        return LocalEditOutput(image=quantize_image(image, params), extra={})
    if op == "slice_grid":
        return LocalEditOutput(image=None, extra={"slices": slice_grid(image, params)})
    if op == "nine_slice_meta":
        return LocalEditOutput(image=image, extra=nine_slice_meta(params))
    raise ValueError(f"Unsupported local edit op: {op}")
