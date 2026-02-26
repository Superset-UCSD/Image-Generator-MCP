from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from PIL import Image

ASSET_ID_RE = re.compile(r"^[a-f0-9]{16}$")


def _normalize_number(value: float) -> float:
    # Keep float representation stable for canonicalization.
    return float(f"{value:.10f}".rstrip("0").rstrip(".")) if "." in f"{value:.10f}" else value


def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: canonicalize(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [canonicalize(v) for v in value]
    if isinstance(value, float):
        return _normalize_number(value)
    return value


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        canonicalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def asset_id_from_payload(payload: dict[str, Any]) -> str:
    canon = canonical_json(payload).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_asset_id(asset_id: str) -> str:
    if not ASSET_ID_RE.fullmatch(asset_id):
        raise ValueError(f"Invalid asset_id: {asset_id}")
    return asset_id


def _resolve_within(base: Path, *parts: str) -> Path:
    candidate = (base.joinpath(*parts)).resolve()
    base_resolved = base.resolve()
    if not str(candidate).startswith(str(base_resolved)):
        raise ValueError("Path traversal is not allowed")
    return candidate


class AssetStore:
    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir).resolve()
        self.images_dir = self.out_dir / "images"
        self.meta_dir = self.out_dir / "meta"
        self.manifests_dir = self.out_dir / "manifests"
        self.manifest_path = self.manifests_dir / "manifest.jsonl"

    def ensure_dirs(self) -> None:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def image_path(self, asset_id: str) -> Path:
        aid = validate_asset_id(asset_id)
        return _resolve_within(self.images_dir, f"{aid}.png")

    def meta_path(self, asset_id: str) -> Path:
        aid = validate_asset_id(asset_id)
        return _resolve_within(self.meta_dir, f"{aid}.json")

    def save_image(self, asset_id: str, image: Image.Image) -> Path:
        self.ensure_dirs()
        out = self.image_path(asset_id)
        image.save(out, format="PNG")
        return out

    def has_image(self, asset_id: str) -> bool:
        return self.image_path(asset_id).exists()

    def load_image(self, asset_id: str) -> Image.Image:
        path = self.image_path(asset_id)
        return Image.open(path)

    def save_meta(self, asset_id: str, meta: dict[str, Any]) -> Path:
        self.ensure_dirs()
        out = self.meta_path(asset_id)
        out.write_text(
            json.dumps(meta, ensure_ascii=True, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return out

    def load_meta(self, asset_id: str) -> dict[str, Any]:
        path = self.meta_path(asset_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cast(dict[str, Any], data)

    def append_manifest(self, record: dict[str, Any]) -> None:
        self.ensure_dirs()
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")

    def append_tool_call(self, tool_name: str, args: dict[str, Any]) -> None:
        entry = {
            "event": "tool_call",
            "tool": tool_name,
            "args": args,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self.append_manifest(entry)

    def manifest_tail(self, n: int = 100) -> str:
        if not self.manifest_path.exists():
            return ""
        lines = self.manifest_path.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-n:])

    def list_assets(
        self,
        limit: int = 50,
        tag: str | None = None,
        task: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.meta_dir.exists():
            return []
        entries: list[dict[str, Any]] = []
        for path in sorted(self.meta_dir.glob("*.json"), reverse=True):
            data = json.loads(path.read_text(encoding="utf-8"))
            if tag and tag not in data.get("tags", []):
                continue
            if task and data.get("task") != task:
                continue
            aid = data["asset_id"]
            entries.append(
                {
                    "asset_id": aid,
                    "created_at": data.get("created_at"),
                    "task": data.get("task"),
                    "tags": data.get("tags", []),
                    "path": str(self.image_path(aid)),
                    "resource_uri": f"asset://image/{aid}",
                    "meta_uri": f"asset://meta/{aid}",
                }
            )
            if len(entries) >= limit:
                break
        return entries


def build_asset_meta(
    *,
    asset_id: str,
    task: str,
    model: str | None,
    params: dict[str, Any],
    image_path: Path,
    input_hash: str | None,
    tags: list[str],
) -> dict[str, Any]:
    return {
        "asset_id": asset_id,
        "created_at": datetime.now(UTC).isoformat(),
        "task": task,
        "model": model,
        "params": params,
        "png_sha256": file_sha256(image_path),
        "input_image_hash": input_hash,
        "tags": sorted(tags),
    }
