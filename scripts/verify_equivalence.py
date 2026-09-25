"""Numerically compare the UI callable against the ORIGINAL archived run_r13.py.

Usage: python scripts/verify_equivalence.py --archive /path/to/master.zip
Only the original canonical Python files and checkpoints are extracted to a
temporary directory. No generating rules or targets enter the inference input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.engine import SEEDS, infer, verify_integrity  # noqa: E402
from runtime.schema import edge_pairs, validate_scenario  # noqa: E402
from runtime.ui_adapter import predict_streamlit_payload  # noqa: E402


def _close(actual: list[float], expected: list[float], label: str, atol: float = 2e-6) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"{label}: size {len(actual)} != {len(expected)}")
    if any(abs(x - y) > atol for x, y in zip(actual, expected)):
        differences = [abs(x - y) for x, y in zip(actual, expected)]
        raise AssertionError(f"{label}: max abs difference {max(differences)} > {atol}")


def extract_reference(z: zipfile.ZipFile, folder: Path) -> None:
    """Direct archived reference, independent from packaged engine imports."""
    (folder / "src").mkdir()
    (folder / "checkpoints").mkdir()
    filenames = [
        "run_r13.py", "src/__init__.py", "src/factor_model.py",
        "src/separated_system_id.py", "src/distilled_evidence_system_id.py",
    ] + [f"checkpoints/R12A_seed{s}_step5000.pt" for s in SEEDS] + [
        f"checkpoints/R12B_seed{s}.pt" for s in SEEDS
    ]
    base = "/VALIDATED_ROLLOUT_R13/FROZEN_PACKAGE/"
    for relative in filenames:
        paths = [name for name in z.namelist() if name.endswith(base + relative)]
        if len(paths) != 1:
            if relative == "src/__init__.py" and len(paths) == 0:
                (folder / relative).write_text("", encoding="utf-8")
                continue
            raise ValueError(f"Missing or ambiguous canonical file: {relative}")
        with z.open(paths[0]) as source, (folder / relative).open("wb") as target:
            shutil.copyfileobj(source, target)
    hashes = verify_integrity()
    for relative in filenames:
        if relative == "run_r13.py" or relative == "src/__init__.py":
            continue
        key = relative.replace("src/", "runtime/canonical/").replace("checkpoints/", "weights/")
        if hashlib.sha256((folder / relative).read_bytes()).hexdigest() != hashes[key]:
            raise AssertionError(f"Packaged file differs from master archive: {relative}")


def load_reference(folder: Path):
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("original_archived_run_r13", folder / "run_r13.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_examples_against_archive(z: zipfile.ZipFile) -> None:
    provenance = json.loads((ROOT / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))
    candidates = [name for name in z.namelist() if name.endswith(
        "/VALIDATED_ROLLOUT_R13/FROZEN_PACKAGE/data/rollout_fresh.jsonl"
    )]
    if len(candidates) != 1:
        raise ValueError("Original R13 observations are missing or ambiguous")
    with z.open(candidates[0]) as f:
        original = {row["row_id"]: row for row in (json.loads(line) for line in f)}
    for entry in provenance["examples"]:
        stored = validate_scenario(json.loads((ROOT / entry["file"]).read_text(encoding="utf-8")))
        row = original[entry["archive_row_id"]]
        if (stored["n_nodes"] != row["n_nodes"]
                or stored["history"] != row["history"]
                or stored["recorded"] != row["rollout"]):
            raise AssertionError(f"Example differs from original observed row: {entry['file']}")


def compare_case(reference, scenario: dict, seed: int, actions: list[int], decoder, belief) -> None:
    n = scenario["n_nodes"]
    history = scenario["history"]
    before_n, before_e = history[-1]["nodes_after"], history[-1]["edges_after"]
    ui = predict_streamlit_payload(json.dumps(history, ensure_ascii=False), tuple(actions), seed)
    # The Streamlit adapter must be byte-for-byte identical to headless public inference.
    if ui != infer(history, before_n, before_e, actions, seed):
        raise AssertionError("UI adapter differs from direct packaged inference")

    # Legacy row_tensor requires labels and rollout, though q_learned reads only h_*.
    # Dummy labels and dummy recorded transitions cannot influence this evaluation.
    fake = {"edge_factor": 0, "node_factor": 0, "n_nodes": n,
            "history": history, "rollout": [history[-1]] * 16}
    batch = reference.collate([reference.row_tensor(fake)])
    qe, qn, q = reference.q_learned(belief, batch, torch.device("cpu"))
    _close(ui["edge_belief"], qe[0].tolist(), "edge belief")
    _close(ui["node_belief"], qn[0].tolist(), "node belief")
    _close(sum(ui["joint_belief"], []), q[0].tolist(), "joint belief")

    pairs = edge_pairs(n)
    mask = batch["h_node_mask"][:, 0]
    valid = reference.valid_mask(mask)
    nodes = batch["h_nodes_after"][:, -1].clone()
    edges = batch["h_edges_after"][:, -1].clone()
    with torch.no_grad():
        for i, action_index in enumerate(actions):
            action = torch.zeros(1, 10)
            action[0, action_index] = 1
            np, ep = reference.mixed_probs(decoder, nodes, edges, action, mask, q)
            nn = (np >= 0.5).float() * mask
            ee = (ep >= 0.5).float() * valid
            step = ui["steps"][i]
            _close(step["nodes_prob"], np[0, :n].tolist(), f"step {i} nodes_prob")
            _close(step["edges_prob"], [float(ep[0, a, b]) for a, b in pairs], f"step {i} edges_prob")
            if step["nodes_after"] != nn[0, :n].int().tolist():
                raise AssertionError(f"step {i}: node state differs")
            if step["edges_after"] != [int(ee[0, a, b]) for a, b in pairs]:
                raise AssertionError(f"step {i}: edge state differs")
            nodes, edges = nn, ee


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path, help="Original R13 master snapshot ZIP")
    args = parser.parse_args()
    verify_integrity()
    with zipfile.ZipFile(args.archive) as z, tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        extract_reference(z, folder)
        verify_examples_against_archive(z)
        reference = load_reference(folder)
        original_models = {
            seed: (reference.load_decoder(seed, torch.device("cpu")),
                   reference.load_sysid(seed, torch.device("cpu"))) for seed in SEEDS
        }
        cases = 0
        for example in sorted((ROOT / "examples").glob("*.json")):
            scenario = validate_scenario(json.loads(example.read_text(encoding="utf-8")))
            for seed in SEEDS:
                decoder, belief = original_models[seed]
                # Original recorded actions and an independently chosen action.
                compare_case(reference, scenario, seed, [t["action"] for t in scenario["recorded"][:3]], decoder, belief)
                compare_case(reference, scenario, seed, [0], decoder, belief)
                cases += 2
        print(f"PASS: {cases} cases; 4 station examples × 3 seeds × 2 action plans; "
              "posterior, all probabilities and hard rollout states match original run_r13.py")


if __name__ == "__main__":
    main()
