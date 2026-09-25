"""Public observation format; no latent factors are accepted from a scenario."""

from __future__ import annotations

from typing import Any


FIELDS = {"action", "nodes_before", "nodes_after", "edges_before", "edges_after"}


def _bits(value: Any, length: int, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label}: expected {length} binary values")
    if any(type(bit) is not int or bit not in (0, 1) for bit in value):
        raise ValueError(f"{label}: values must be integers 0 or 1")
    return value


def validate_state(nodes: Any, edges: Any, n: int) -> None:
    _bits(nodes, n, "nodes")
    _bits(edges, n * (n - 1), "edges")


def validate_history(history: Any, n: int) -> None:
    if not isinstance(history, list) or len(history) != 8:
        raise ValueError("History must contain exactly eight observed transitions")
    _transitions(history, n, "history")


def _transitions(transitions: list[dict], n: int, label: str) -> None:
    previous = None
    for i, t in enumerate(transitions):
        if not isinstance(t, dict) or set(t) != FIELDS:
            raise ValueError(f"{label}[{i}]: unexpected or missing fields")
        if type(t["action"]) is not int or not 0 <= t["action"] < n:
            raise ValueError(f"{label}[{i}]: action must be a node index")
        for name in ("nodes_before", "nodes_after"):
            _bits(t[name], n, f"{label}[{i}].{name}")
        for name in ("edges_before", "edges_after"):
            _bits(t[name], n * (n - 1), f"{label}[{i}].{name}")
        if previous is not None and (
            previous["nodes_after"] != t["nodes_before"]
            or previous["edges_after"] != t["edges_before"]
        ):
            raise ValueError(f"{label}[{i}]: transition does not follow its predecessor")
        previous = t


def validate_scenario(scenario: Any) -> dict:
    if not isinstance(scenario, dict) or not {"n_nodes", "history"} <= set(scenario):
        raise ValueError("Scenario requires n_nodes and history")
    if set(scenario) - {"name", "n_nodes", "history", "recorded"}:
        raise ValueError("Scenario contains hidden or unsupported fields")
    n = scenario["n_nodes"]
    if type(n) is not int or not 7 <= n <= 10:
        raise ValueError("The frozen station runtime supports 7–10 nodes")
    if "name" in scenario and (not isinstance(scenario["name"], str) or len(scenario["name"]) > 100):
        raise ValueError("Invalid name")
    validate_history(scenario["history"], n)
    recorded = scenario.get("recorded", [])
    if not isinstance(recorded, list) or len(recorded) > 16:
        raise ValueError("At most 16 recorded future transitions are supported")
    _transitions(recorded, n, "recorded")
    if recorded and (
        recorded[0]["nodes_before"] != scenario["history"][-1]["nodes_after"]
        or recorded[0]["edges_before"] != scenario["history"][-1]["edges_after"]
    ):
        raise ValueError("Recorded future does not follow observed history")
    return scenario


def observed_next(scenario: dict, index: int, nodes: list[int], edges: list[int], action: int) -> dict | None:
    """Only an existing recorded transition can be called a real observation."""
    recorded = scenario.get("recorded", [])
    if index >= len(recorded):
        return None
    t = recorded[index]
    if t["nodes_before"] == nodes and t["edges_before"] == edges and t["action"] == action:
        return t
    return None


def edge_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def changes(before: list[int], after: list[int], labels: list[str]) -> list[str]:
    return [f"{label}: {a} → {b}" for label, a, b in zip(labels, before, after) if a != b]
