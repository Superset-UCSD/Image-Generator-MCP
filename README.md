# image-gen-mcp

`image-gen-mcp` is a generic placeholder image generator and editor MCP server for Codex clients.
It supports Hugging Face text-to-image and image-to-image generation, deterministic local Pillow edits, and MCP resources for direct image/meta access.

Initially built this to easily iterate through and generate placeholder images for game development and any other related.

## Features

- Text-to-image generation (`image.generate`)
- AI image editing (`image.ai_edit`)
- Deterministic local edits (`image.local_edit`): resize, crop, pad, rotate, flip, convert, quantize, slice grid, nine-slice metadata
- Batch execution (`image.batch`)
- Asset listing and metadata lookup (`image.list`, `image.get_meta`)
- Health checks (`image.healthcheck`)
- MCP resources:
  - `asset://image/<asset_id>`: PNG bytes
  - `asset://meta/<asset_id>`: metadata JSON
  - `asset://manifest/latest?n=100`: latest manifest JSONL lines
- Deterministic asset IDs and on-disk cache

## Install and Run (uv)

```bash
uv sync
uv run image-gen --set-api hf_xxx
uv run image-gen doctor
uv run image-gen setup-codex
uv run image-gen-mcp
```

## Token Configuration

Set your Hugging Face token in one command:

```bash
uv run image-gen --set-api hf_xxx
```

Token precedence at runtime:

1. `--api-key` CLI option (per run)
2. `HF_TOKEN` environment variable
3. stored config (`config.toml`)

Token values are redacted in output.

## CLI Commands

```bash
uv run image-gen --set-api hf_xxx
uv run image-gen config show
uv run image-gen config path
uv run image-gen config set-defaults --t2i-model stabilityai/stable-diffusion-xl-base-1.0 --size 1024x1024 --steps 28 --guidance 7.0 --max-concurrency 2
uv run image-gen doctor
uv run image-gen doctor --smoke
uv run image-gen setup-codex
```

## MCP Tool Examples

`image.generate`:
The below is an example prompt with generic known tags that work well.

```json
{
  "name": "image.generate",
  "arguments": {
    "prompt": "masterpiece, best quality, cartoon, portrait, stylistic lines, neutral lighting",
    "width": 512,
    "height": 512,
    "tags": ["placeholder", "portrait"]
  }
}
```

`image.ai_edit`:

```json
{
  "name": "image.ai_edit",
  "arguments": {
    "input_asset_id": "abcd1234ef567890",
    "prompt": "Refine details and add subtle rim light",
    "strength": 0.6
  }
}
```

`image.local_edit`:

```json
{
  "name": "image.local_edit",
  "arguments": {
    "input_asset_id": "abcd1234ef567890",
    "op": "resize",
    "params": {"width": 512, "height": 512, "mode": "contain", "background": "#00000000"}
  }
}
```

`image.batch`:

```json
{
  "name": "image.batch",
  "arguments": {
    "max_parallel": 2,
    "items": [
      {"tool": "image.generate", "args": {"prompt": "Simple icon placeholder"}},
      {"tool": "image.generate", "args": {"prompt": "Simple environment placeholder"}}
    ]
  }
}
```

## Resource URIs

- `asset://image/<asset_id>` returns binary PNG bytes.
- `asset://meta/<asset_id>` returns JSON metadata.
- `asset://manifest/latest?n=100` returns the last `n` JSONL entries.

Clients can fetch these resources directly after receiving `resource_uri`/`meta_uri` from tool responses.

## Output, Metadata, and Caching

Assets are stored under `out_dir` (default `./assets_out`):

- `images/<asset_id>.png`
- `meta/<asset_id>.json`
- `manifests/manifest.jsonl`

Asset IDs are deterministic:

- `asset_id = sha256(canonical_json(request_params)).hexdigest()[:16]`

If an image for the same canonical request already exists and `force=false`, tools return a cache hit.

Each new asset writes metadata JSON and appends one JSON line to the manifest.

## Security Boundaries

- File reads/writes are restricted to configured `out_dir` subdirectories.
- Path traversal is rejected.
- No shell execution and no arbitrary URL downloads.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

Network smoke tests are disabled by default and only run when explicitly requested.

## License

MIT. See [LICENSE](LICENSE).

## Contributing

1. Create a branch.
2. Add tests for behavior changes.
3. Run `uv run pytest`, `uv run ruff check .`, and `uv run mypy src`.
4. Open a pull request.
