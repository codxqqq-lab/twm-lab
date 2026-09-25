"""Frozen R13 inference: learned R12 belief + neural decoder, CPU only.

Tensorization, 36-pair mixture and 0.5 hard rollout mirror run_r13.py.
The model never sees recorded futures or hidden factor labels.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from runtime.canonical.factor_model import FactorizedRelationalTransitionModel
from runtime.canonical.separated_system_id import FactorSeparatedSystemID
from runtime.canonical.distilled_evidence_system_id import DistilledNodeEvidence
from runtime.schema import edge_pairs, validate_history, validate_state


ROOT = Path(__file__).resolve().parents[1]
MAX_N = 10
SEEDS = (17, 29, 43)
PAIRS = [(e, n) for e in range(6) for n in range(6)]


class R12SystemID(nn.Module):
    def __init__(self, edge: nn.Module, node: nn.Module):
        super().__init__()
        self.edge = edge
        self.node = node

    def forward(self, hist: dict) -> dict:
        return {"edge_logits": self.edge(hist), "node_logits": self.node(hist)}


def verify_integrity() -> dict[str, str]:
    expected = json.loads((ROOT / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))["sha256"]
    for name, digest in expected.items():
        path = ROOT / name
        with path.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != digest:
            raise RuntimeError(f"Canonical file has changed: {name}")
    return expected


def _verified_checkpoint(name: str) -> dict:
    expected = json.loads((ROOT / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))["sha256"]
    path = ROOT / "weights" / name
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != expected[f"weights/{name}"]:
            raise RuntimeError(f"Checkpoint SHA-256 mismatch: {name}")
    # This exact archived torch checkpoint requires PyTorch's pickle loader.
    # Digest verification happens before any checkpoint is deserialized.
    return torch.load(path, map_location="cpu", weights_only=False)


@lru_cache(maxsize=3)
def load_models(seed: int) -> tuple[nn.Module, nn.Module]:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    verify_integrity()
    decoder = FactorizedRelationalTransitionModel(nedge=6, nnode=6)
    decoder.load_state_dict(_verified_checkpoint(f"R12A_seed{seed}_step5000.pt")["model"])
    decoder.eval()
    base = FactorSeparatedSystemID()
    belief = R12SystemID(base.edge, DistilledNodeEvidence())
    belief.load_state_dict(_verified_checkpoint(f"R12B_seed{seed}.pt")["model"])
    belief.eval()
    for model in (decoder, belief):
        for param in model.parameters():
            param.requires_grad_(False)
    return decoder, belief


def _matrix(flat: list[int], n: int) -> torch.Tensor:
    result = torch.zeros(MAX_N, MAX_N, dtype=torch.float32)
    for value, (i, j) in zip(flat, edge_pairs(n)):
        result[i, j] = float(value)
    return result


def _history_inputs(history: list[dict], n: int) -> dict[str, torch.Tensor]:
    nm = torch.zeros(MAX_N)
    nm[:n] = 1
    inputs: dict[str, list[torch.Tensor]] = {
        name: [] for name in (
            "nodes_before", "edges_before", "action", "node_mask", "nodes_after", "edges_after"
        )
    }
    for t in history:
        nb, na = torch.zeros(MAX_N), torch.zeros(MAX_N)
        nb[:n] = torch.tensor(t["nodes_before"], dtype=torch.float32)
        na[:n] = torch.tensor(t["nodes_after"], dtype=torch.float32)
        action = torch.zeros(MAX_N)
        action[t["action"]] = 1
        values = {
            "nodes_before": nb, "edges_before": _matrix(t["edges_before"], n),
            "action": action, "node_mask": nm,
            "nodes_after": na, "edges_after": _matrix(t["edges_after"], n),
        }
        for name in inputs:
            inputs[name].append(values[name])
    return {name: torch.stack(seq)[None] for name, seq in inputs.items()}


def _all_factor_probs(decoder: nn.Module, nodes: torch.Tensor, edges: torch.Tensor,
                      action: torch.Tensor, node_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Exactly the frozen 36-pair batched decoder path from run_r13.py.
    batch, width = nodes.shape
    factor_count = 36
    edge_ids = torch.tensor([e for e, _ in PAIRS], dtype=torch.long, device=nodes.device)
    node_ids = torch.tensor([g for _, g in PAIRS], dtype=torch.long, device=nodes.device)

    def expand(x: torch.Tensor, tail: tuple[int, ...]) -> torch.Tensor:
        return x[:, None].expand((batch, factor_count) + tail).reshape((batch * factor_count,) + tail)

    inp = {
        "nodes": expand(nodes, (width,)), "edges": expand(edges, (width, width)),
        "action": expand(action, (width,)), "node_mask": expand(node_mask, (width,)),
        "edge_factor": edge_ids[None, :].expand(batch, factor_count).reshape(-1),
        "node_factor": node_ids[None, :].expand(batch, factor_count).reshape(-1),
    }
    output = decoder(inp)
    return (output["next_nodes_prob"].reshape(batch, factor_count, width),
            output["next_edges_prob"].reshape(batch, factor_count, width, width))


@torch.no_grad()
def infer(history: list[dict], nodes: list[int], edges: list[int], actions: list[int], seed: int = 17) -> dict:
    """Predict one or more actions; predicted steps do not become observed history."""
    n = len(nodes)
    if n < 7 or n > MAX_N:
        raise ValueError("n_nodes must be between 7 and 10")
    validate_history(history, n)
    validate_state(nodes, edges, n)
    if not isinstance(actions, list) or not 1 <= len(actions) <= 16 or any(
        type(a) is not int or not 0 <= a < n for a in actions
    ):
        raise ValueError("Provide 1–16 node-index actions")
    decoder, belief = load_models(seed)
    h = _history_inputs(history, n)
    edge_logits = belief.edge(h)
    node_logits = belief.node(h)
    qe = torch.softmax(edge_logits, -1)
    qn = torch.softmax(node_logits, -1)
    q = (qe[:, :, None] * qn[:, None, :]).reshape(1, 36)
    current_nodes = torch.zeros(1, MAX_N)
    current_nodes[0, :n] = torch.tensor(nodes, dtype=torch.float32)
    current_edges = _matrix(edges, n)[None]
    mask = torch.zeros(1, MAX_N)
    mask[0, :n] = 1
    valid = (mask[:, :, None] * mask[:, None, :]) * (1 - torch.eye(MAX_N)[None])
    pairs = edge_pairs(n)
    steps = []
    for a in actions:
        action = torch.zeros(1, MAX_N)
        action[0, a] = 1
        node_probs_by_pair, edge_probs_by_pair = _all_factor_probs(
            decoder, current_nodes, current_edges, action, mask
        )
        node_prob = (node_probs_by_pair * q[:, :, None]).sum(1)
        edge_prob = (edge_probs_by_pair * q[:, :, None, None]).sum(1)
        predicted_nodes = (node_prob >= 0.5).float() * mask
        predicted_edges = (edge_prob >= 0.5).float() * valid
        before_nodes = current_nodes[0, :n].int().tolist()
        before_edges = [int(current_edges[0, i, j]) for i, j in pairs]
        after_nodes = predicted_nodes[0, :n].int().tolist()
        after_edges = [int(predicted_edges[0, i, j]) for i, j in pairs]
        steps.append({
            "action": a, "nodes_before": before_nodes, "edges_before": before_edges,
            "nodes_prob": node_prob[0, :n].tolist(),
            "edges_prob": [float(edge_prob[0, i, j]) for i, j in pairs],
            "nodes_after": after_nodes, "edges_after": after_edges,
        })
        current_nodes, current_edges = predicted_nodes, predicted_edges
    return {
        "checkpoint_seed": seed,
        "edge_logits": edge_logits[0].tolist(), "node_logits": node_logits[0].tolist(),
        "edge_belief": qe[0].tolist(), "node_belief": qn[0].tolist(),
        "joint_belief": q[0].reshape(6, 6).tolist(),
        "steps": steps,
    }
