from __future__ import annotations

import json

import typer
from dotenv import load_dotenv

from .config import (
    config_path,
    effective_config,
    set_defaults,
    set_token,
    shown_config,
)
from .mcp_tools import ImageGenService

load_dotenv()

app = typer.Typer(help="Image generation and editing CLI")
config_app = typer.Typer(help="Manage configuration")
app.add_typer(config_app, name="config")


class State:
    def __init__(self) -> None:
        self.api_key: str | None = None


state = State()


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    set_api: str | None = typer.Option(None, "--set-api", help="Set HF token in config"),
    api_key: str | None = typer.Option(None, "--api-key", help="HF token override for this run"),
) -> None:
    state.api_key = api_key
    if set_api:
        path = set_token(set_api)
        typer.echo(f"Stored HF token in {path}")
        if ctx.invoked_subcommand is None:
            raise typer.Exit(0)


@config_app.command("show")
def config_show() -> None:
    cfg = effective_config(state.api_key)
    typer.echo(json.dumps(shown_config(cfg), indent=2, sort_keys=True))


@config_app.command("path")
def config_show_path() -> None:
    typer.echo(str(config_path()))


@config_app.command("set-defaults")
def config_set_defaults(
    t2i_model: str | None = typer.Option(None),
    i2i_model: str | None = typer.Option(None),
    out_dir: str | None = typer.Option(None),
    size: str | None = typer.Option(None, help="Format: WIDTHxHEIGHT"),
    steps: int | None = typer.Option(None),
    guidance: float | None = typer.Option(None),
    max_concurrency: int | None = typer.Option(None),
) -> None:
    width: int | None = None
    height: int | None = None
    if size:
        parts = size.lower().split("x")
        if len(parts) != 2:
            raise typer.BadParameter("size must be WIDTHxHEIGHT")
        width = int(parts[0])
        height = int(parts[1])

    cfg = set_defaults(
        t2i_model=t2i_model,
        i2i_model=i2i_model,
        out_dir=out_dir,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        max_concurrency=max_concurrency,
    )
    typer.echo(json.dumps(shown_config(cfg), indent=2, sort_keys=True))


@app.command("doctor")
def doctor(
    smoke: bool = typer.Option(False, "--smoke", help="Run tiny generation smoke test"),
) -> None:
    service = ImageGenService(effective_config(state.api_key))
    result = service.healthcheck(smoke=smoke)
    if result.get("has_token"):
        result["has_token"] = True
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
