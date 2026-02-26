from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .cache import AssetStore, asset_id_from_payload, build_asset_meta, bytes_sha256
from .config import AppConfig, effective_config
from .hf_client import HFImageClient
from .image_ops import run_local_edit
from .models import AiEditArgs, BatchArgs, GenerateArgs, LocalEditArgs


class ImageGenService:
    def __init__(self, cfg: AppConfig | None = None, *, api_key: str | None = None) -> None:
        self.cfg = cfg or effective_config(api_key)
        self.store = AssetStore(self.cfg.out_dir)
        self.store.ensure_dirs()
        self.hf_client = HFImageClient(self.cfg) if self.cfg.hf_token else None

    def _resource_paths(self, asset_id: str) -> tuple[str, str]:
        return f"asset://image/{asset_id}", f"asset://meta/{asset_id}"

    def _asset_result(
        self,
        *,
        asset_id: str,
        image: Image.Image,
        model: str | None,
        prompt: str | None,
        negative_prompt: str | None,
        cache_hit: bool,
        input_image_hash: str | None = None,
    ) -> dict[str, Any]:
        resource_uri, meta_uri = self._resource_paths(asset_id)
        return {
            "asset_id": asset_id,
            "path": str(self.store.image_path(asset_id)),
            "mime_type": "image/png",
            "width": image.width,
            "height": image.height,
            "model": model,
            "prompt_used": prompt,
            "negative_prompt_used": negative_prompt,
            "cache_hit": cache_hit,
            "resource_uri": resource_uri,
            "meta_uri": meta_uri,
            "input_image_hash": input_image_hash,
        }

    def _save_new_asset(
        self,
        *,
        asset_id: str,
        image: Image.Image,
        task: str,
        model: str | None,
        params: dict[str, Any],
        input_hash: str | None,
        tags: list[str],
    ) -> dict[str, Any]:
        image_path = self.store.save_image(asset_id, image)
        meta = build_asset_meta(
            asset_id=asset_id,
            task=task,
            model=model,
            params=params,
            image_path=image_path,
            input_hash=input_hash,
            tags=tags,
        )
        self.store.save_meta(asset_id, meta)
        self.store.append_manifest({"event": "asset", **meta})
        return meta

    def _resolve_input_image(
        self,
        input_asset_id: str | None,
        input_path: str | None,
    ) -> tuple[Image.Image, str]:
        if bool(input_asset_id) == bool(input_path):
            raise ValueError("Provide exactly one of input_asset_id or input_path")

        if input_asset_id:
            image_path = self.store.image_path(input_asset_id)
        else:
            candidate = Path(str(input_path)).expanduser().resolve()
            out = Path(self.store.out_dir).resolve()
            if not str(candidate).startswith(str(out)):
                raise ValueError("input_path must be inside out_dir")
            image_path = candidate

        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        data = image_path.read_bytes()
        return Image.open(image_path).convert("RGBA"), bytes_sha256(data)

    def generate(self, args: GenerateArgs) -> dict[str, Any]:
        self.store.append_tool_call("image.generate", args.model_dump(mode="json"))
        model = args.model or self.cfg.defaults.t2i_model
        width = args.width or self.cfg.defaults.width
        height = args.height or self.cfg.defaults.height
        steps = args.steps or self.cfg.defaults.steps
        guidance = args.guidance if args.guidance is not None else self.cfg.defaults.guidance
        negative = (
            args.negative_prompt
            if args.negative_prompt is not None
            else self.cfg.defaults.negative_prompt
        )
        tags = sorted(args.tags or [])

        payload = {
            "task": "t2i",
            "model": model,
            "prompt": args.prompt,
            "negative_prompt": negative,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "seed": args.seed,
            "strength": None,
            "input_image_hash": None,
            "local_edit": None,
            "tags": tags,
        }
        asset_id = asset_id_from_payload(payload)

        if self.store.has_image(asset_id) and not args.force:
            image = self.store.load_image(asset_id)
            if not self.store.meta_path(asset_id).exists():
                self._save_new_asset(
                    asset_id=asset_id,
                    image=image,
                    task="t2i",
                    model=model,
                    params=payload,
                    input_hash=None,
                    tags=tags,
                )
            return self._asset_result(
                asset_id=asset_id,
                image=image,
                model=model,
                prompt=args.prompt,
                negative_prompt=negative,
                cache_hit=True,
            )

        if self.hf_client is None:
            raise RuntimeError("HF token missing. Configure with image-gen --set-api or HF_TOKEN")

        image = self.hf_client.text_to_image(
            prompt=args.prompt,
            negative_prompt=negative,
            model=model,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=args.seed,
        ).convert("RGBA")

        self._save_new_asset(
            asset_id=asset_id,
            image=image,
            task="t2i",
            model=model,
            params=payload,
            input_hash=None,
            tags=tags,
        )
        return self._asset_result(
            asset_id=asset_id,
            image=image,
            model=model,
            prompt=args.prompt,
            negative_prompt=negative,
            cache_hit=False,
        )

    def ai_edit(self, args: AiEditArgs) -> dict[str, Any]:
        self.store.append_tool_call("image.ai_edit", args.model_dump(mode="json"))
        model = args.model or self.cfg.defaults.i2i_model
        steps = args.steps or self.cfg.defaults.steps
        guidance = args.guidance if args.guidance is not None else self.cfg.defaults.guidance
        negative = (
            args.negative_prompt
            if args.negative_prompt is not None
            else self.cfg.defaults.negative_prompt
        )
        tags = sorted(args.tags or [])

        input_image, input_hash = self._resolve_input_image(args.input_asset_id, args.input_path)
        payload = {
            "task": "i2i",
            "model": model,
            "prompt": args.prompt,
            "negative_prompt": negative,
            "width": input_image.width,
            "height": input_image.height,
            "steps": steps,
            "guidance": guidance,
            "seed": args.seed,
            "strength": args.strength,
            "input_image_hash": input_hash,
            "local_edit": None,
            "tags": tags,
        }
        asset_id = asset_id_from_payload(payload)

        if self.store.has_image(asset_id) and not args.force:
            image = self.store.load_image(asset_id)
            return self._asset_result(
                asset_id=asset_id,
                image=image,
                model=model,
                prompt=args.prompt,
                negative_prompt=negative,
                cache_hit=True,
                input_image_hash=input_hash,
            )

        if self.hf_client is None:
            raise RuntimeError("HF token missing. Configure with image-gen --set-api or HF_TOKEN")

        image = self.hf_client.image_to_image(
            input_image=input_image,
            prompt=args.prompt,
            negative_prompt=negative,
            model=model,
            steps=steps,
            guidance=guidance,
            seed=args.seed,
            strength=args.strength or 0.6,
        ).convert("RGBA")

        self._save_new_asset(
            asset_id=asset_id,
            image=image,
            task="i2i",
            model=model,
            params=payload,
            input_hash=input_hash,
            tags=tags,
        )
        return self._asset_result(
            asset_id=asset_id,
            image=image,
            model=model,
            prompt=args.prompt,
            negative_prompt=negative,
            cache_hit=False,
            input_image_hash=input_hash,
        )

    def local_edit(self, args: LocalEditArgs) -> dict[str, Any]:
        self.store.append_tool_call("image.local_edit", args.model_dump(mode="json"))
        input_image, input_hash = self._resolve_input_image(args.input_asset_id, args.input_path)
        tags = sorted(args.tags or [])

        payload = {
            "task": "local_edit",
            "model": None,
            "prompt": None,
            "negative_prompt": None,
            "width": input_image.width,
            "height": input_image.height,
            "steps": None,
            "guidance": None,
            "seed": None,
            "strength": None,
            "input_image_hash": input_hash,
            "local_edit": {"op": args.op, "params": args.params},
            "tags": tags,
        }

        output = run_local_edit(input_image, args.op, args.params, _out_dir=self.store.out_dir)

        if args.op == "slice_grid":
            results: list[dict[str, Any]] = []
            for idx, image in enumerate(output.extra["slices"]):
                item_payload = {
                    **payload,
                    "local_edit": {"op": args.op, "params": args.params, "index": idx},
                }
                aid = asset_id_from_payload(item_payload)
                if self.store.has_image(aid) and not args.force:
                    cached = self.store.load_image(aid)
                    results.append(
                        self._asset_result(
                            asset_id=aid,
                            image=cached,
                            model=None,
                            prompt=None,
                            negative_prompt=None,
                            cache_hit=True,
                            input_image_hash=input_hash,
                        )
                    )
                    continue

                self._save_new_asset(
                    asset_id=aid,
                    image=image,
                    task="local_edit",
                    model=None,
                    params=item_payload,
                    input_hash=input_hash,
                    tags=tags,
                )
                results.append(
                    self._asset_result(
                        asset_id=aid,
                        image=image,
                        model=None,
                        prompt=None,
                        negative_prompt=None,
                        cache_hit=False,
                        input_image_hash=input_hash,
                    )
                )
            return {"results": results}

        aid = asset_id_from_payload(payload)
        if self.store.has_image(aid) and not args.force:
            cached = self.store.load_image(aid)
            if args.op == "nine_slice_meta":
                meta = self.store.load_meta(aid)
                return {"asset_id": aid, "meta": meta, "meta_uri": f"asset://meta/{aid}"}
            return self._asset_result(
                asset_id=aid,
                image=cached,
                model=None,
                prompt=None,
                negative_prompt=None,
                cache_hit=True,
                input_image_hash=input_hash,
            )

        out_img = (output.image or input_image).convert("RGBA")
        meta = self._save_new_asset(
            asset_id=aid,
            image=out_img,
            task="local_edit",
            model=None,
            params=payload,
            input_hash=input_hash,
            tags=tags,
        )
        if args.op == "nine_slice_meta":
            meta["meta"] = output.extra
            self.store.save_meta(aid, meta)
            self.store.append_manifest(
                {
                    "event": "nine_slice_meta",
                    "asset_id": aid,
                    "meta": output.extra,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
            return {"asset_id": aid, "meta": output.extra, "meta_uri": f"asset://meta/{aid}"}

        return self._asset_result(
            asset_id=aid,
            image=out_img,
            model=None,
            prompt=None,
            negative_prompt=None,
            cache_hit=False,
            input_image_hash=input_hash,
        )

    def batch(self, args: BatchArgs) -> dict[str, Any]:
        self.store.append_tool_call("image.batch", args.model_dump(mode="json"))
        results: list[Any] = [None for _ in args.items]

        def run_item(idx: int, item: Any) -> tuple[int, Any]:
            if item.tool == "image.generate":
                return idx, self.generate(GenerateArgs.model_validate(item.args))
            if item.tool == "image.ai_edit":
                return idx, self.ai_edit(AiEditArgs.model_validate(item.args))
            if item.tool == "image.local_edit":
                return idx, self.local_edit(LocalEditArgs.model_validate(item.args))
            raise ValueError(f"Unsupported tool in batch: {item.tool}")

        with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as ex:
            futures = [ex.submit(run_item, idx, item) for idx, item in enumerate(args.items)]
            for future in futures:
                idx, value = future.result()
                results[idx] = value

        return {"results": results}

    def list_assets(
        self,
        *,
        limit: int,
        tag: str | None,
        task: str | None,
    ) -> list[dict[str, Any]]:
        self.store.append_tool_call("image.list", {"limit": limit, "tag": tag, "task": task})
        return self.store.list_assets(limit=limit, tag=tag, task=task)

    def get_meta(self, asset_id: str) -> dict[str, Any]:
        self.store.append_tool_call("image.get_meta", {"asset_id": asset_id})
        return self.store.load_meta(asset_id)

    def healthcheck(self, smoke: bool = False) -> dict[str, Any]:
        self.store.append_tool_call("image.healthcheck", {"smoke": smoke})
        try:
            self.store.ensure_dirs()
            has_token = bool(self.cfg.hf_token)
            if smoke:
                if not has_token:
                    return {
                        "ok": False,
                        "has_token": False,
                        "out_dir": str(self.store.out_dir),
                        "smoke_ran": False,
                        "error": "HF token missing",
                    }
                self.generate(
                    GenerateArgs(
                        prompt="Minimal placeholder scene",
                        width=512,
                        height=512,
                        steps=8,
                        guidance=5.0,
                        tags=["smoke"],
                    )
                )
            return {
                "ok": True,
                "has_token": has_token,
                "out_dir": str(self.store.out_dir),
                "smoke_ran": smoke,
            }
        except Exception as exc:
            return {
                "ok": False,
                "has_token": bool(self.cfg.hf_token),
                "out_dir": str(self.store.out_dir),
                "smoke_ran": smoke,
                "error": str(exc),
            }


def register_tools(mcp: Any, service: ImageGenService) -> None:
    @mcp.tool(name="image.generate")
    def image_generate(args: GenerateArgs) -> dict[str, Any]:
        return service.generate(args)

    @mcp.tool(name="image.ai_edit")
    def image_ai_edit(args: AiEditArgs) -> dict[str, Any]:
        return service.ai_edit(args)

    @mcp.tool(name="image.local_edit")
    def image_local_edit(args: LocalEditArgs) -> dict[str, Any]:
        return service.local_edit(args)

    @mcp.tool(name="image.batch")
    def image_batch(args: BatchArgs) -> dict[str, Any]:
        return service.batch(args)

    @mcp.tool(name="image.list")
    def image_list(
        limit: int = 50,
        tag: str | None = None,
        task: str | None = None,
    ) -> list[dict[str, Any]]:
        return service.list_assets(limit=limit, tag=tag, task=task)

    @mcp.tool(name="image.get_meta")
    def image_get_meta(asset_id: str) -> dict[str, Any]:
        return service.get_meta(asset_id)

    @mcp.tool(name="image.healthcheck")
    def image_healthcheck(smoke: bool = False) -> dict[str, Any]:
        return service.healthcheck(smoke=smoke)
