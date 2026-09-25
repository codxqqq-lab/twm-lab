"""Format-only boundary between the public observation and frozen inference."""

from __future__ import annotations

import json

from runtime.engine import infer


def predict_ui(history: list[dict], actions: list[int], seed: int) -> dict:
    last = history[-1]
    return infer(history, last["nodes_after"], last["edges_after"], actions, seed)


def predict_streamlit_payload(history_json: str, actions: tuple[int, ...], seed: int) -> dict:
    """The exact serialization boundary invoked by the Streamlit cache wrapper."""
    return predict_ui(json.loads(history_json), list(actions), seed)
