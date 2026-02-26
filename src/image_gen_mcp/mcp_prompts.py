from __future__ import annotations

from typing import Any


def register_prompts(mcp: Any) -> None:
    @mcp.prompt(name="prompt.portrait")
    def prompt_portrait(
        subject: str,
        style: str = "clean illustrative",
        lighting: str = "soft studio",
        background: str = "simple gradient",
        composition: str = "head and shoulders centered",
    ) -> dict[str, Any]:
        return {
            "prompt": (
                f"Portrait of {subject}, {style}, {lighting} lighting, "
                f"{background} background, {composition}, high detail, clean edges"
            ),
            "negative_prompt": "low quality, blurry, artifacts, distorted anatomy, text watermark",
            "suggested_params": {"width": 1024, "height": 1024, "steps": 28, "guidance": 7.0},
        }

    @mcp.prompt(name="prompt.icon")
    def prompt_icon(
        object: str,
        style: str = "flat vector-like",
        stroke: str = "thin outline",
        background: str = "transparent",
    ) -> dict[str, Any]:
        return {
            "prompt": (
                f"Icon of {object}, {style}, {stroke}, {background} background, "
                "balanced silhouette, centered composition"
            ),
            "negative_prompt": "photorealistic clutter, blurry edges, watermark, text",
            "suggested_params": {"width": 1024, "height": 1024, "steps": 24, "guidance": 6.5},
        }

    @mcp.prompt(name="prompt.ui_panel")
    def prompt_ui_panel(
        material: str,
        shape_language: str,
        ornament: str,
        texture_level: str,
        readability: str,
    ) -> dict[str, Any]:
        return {
            "prompt": (
                "UI panel concept, "
                f"material {material}, shape language {shape_language}, ornament {ornament}, "
                f"texture level {texture_level}, readability {readability}, clean UI framing"
            ),
            "negative_prompt": "busy composition, illegible layout, noisy texture, watermark",
            "suggested_params": {"width": 1536, "height": 1024, "steps": 30, "guidance": 7.5},
        }

    @mcp.prompt(name="prompt.environment_bg")
    def prompt_environment_bg(
        location: str,
        mood: str,
        time: str,
        palette: str,
    ) -> dict[str, Any]:
        return {
            "prompt": (
                f"Environment background of {location}, mood {mood}, time {time}, "
                f"palette {palette}, layered depth, atmospheric perspective"
            ),
            "negative_prompt": "character close-up, text overlays, logo, low detail",
            "suggested_params": {"width": 1536, "height": 1024, "steps": 30, "guidance": 7.0},
        }
