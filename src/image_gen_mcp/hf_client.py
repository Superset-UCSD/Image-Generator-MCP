from __future__ import annotations

from io import BytesIO
from threading import Semaphore
from typing import Any

from huggingface_hub import InferenceClient
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import AppConfig


class HFClientError(RuntimeError):
    pass


class HFImageClient:
    def __init__(self, cfg: AppConfig) -> None:
        token = cfg.hf_token
        if not token:
            raise HFClientError("Hugging Face token is not configured")
        self.client = InferenceClient(token=token)
        self.sem = Semaphore(cfg.defaults.max_concurrency)
        self.retry_max = cfg.defaults.retry_max
        self.retry_base_delay = cfg.defaults.retry_base_delay

    def _text_to_image_retry(self, **kwargs: Any) -> Image.Image:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.retry_max),
            wait=wait_exponential(
                multiplier=self.retry_base_delay,
                min=self.retry_base_delay,
                max=20,
            ),
            retry=retry_if_exception_type(Exception),
        )
        def _inner() -> Image.Image:
            result = self.client.text_to_image(**kwargs)
            return _ensure_pil(result)

        return _inner()

    def _image_to_image_retry(self, **kwargs: Any) -> Image.Image:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self.retry_max),
            wait=wait_exponential(
                multiplier=self.retry_base_delay,
                min=self.retry_base_delay,
                max=20,
            ),
            retry=retry_if_exception_type(Exception),
        )
        def _inner() -> Image.Image:
            result = self.client.image_to_image(**kwargs)
            return _ensure_pil(result)

        return _inner()

    def _call_text_to_image(self, **kwargs: Any) -> Image.Image:
        result = self._text_to_image_retry(**kwargs)
        return _ensure_pil(result)

    def _call_image_to_image(self, **kwargs: Any) -> Image.Image:
        result = self._image_to_image_retry(**kwargs)
        return _ensure_pil(result)

    def text_to_image(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        model: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int | None,
    ) -> Image.Image:
        with self.sem:
            return self._call_text_to_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model=model,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed,
            )

    def image_to_image(
        self,
        *,
        input_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        model: str,
        steps: int,
        guidance: float,
        seed: int | None,
        strength: float,
    ) -> Image.Image:
        with self.sem:
            return self._call_image_to_image(
                image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model=model,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed,
                strength=strength,
            )


def _ensure_pil(result: object) -> Image.Image:
    if isinstance(result, Image.Image):
        return result
    if isinstance(result, bytes):
        return Image.open(BytesIO(result))
    raise HFClientError(f"Unexpected HF response type: {type(result)}")
