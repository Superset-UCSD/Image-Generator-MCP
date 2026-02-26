from __future__ import annotations

from image_gen_mcp.cache import asset_id_from_payload, canonical_json


def test_canonicalization_deterministic() -> None:
    payload_a = {
        "task": "local_edit",
        "tags": ["b", "a"],
        "local_edit": {"params": {"w": 100, "h": 100}, "op": "resize"},
        "guidance": 7.0,
    }
    payload_b = {
        "guidance": 7.0000,
        "local_edit": {"op": "resize", "params": {"h": 100, "w": 100}},
        "tags": ["b", "a"],
        "task": "local_edit",
    }
    assert canonical_json(payload_a) == canonical_json(payload_b)
    assert asset_id_from_payload(payload_a) == asset_id_from_payload(payload_b)
