"""Deployment-only bootstrap for canonical checkpoints.

The lab repository keeps UI changes separate from model weights. On first run it
downloads the exact public canonical checkpoints from codxqqq-lab/twm, verifies
SHA-256 from SOURCE_PROVENANCE.json, then runtime.engine verifies them again
before torch.load. This module never changes model tensors or inference logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://raw.githubusercontent.com/codxqqq-lab/twm/main"


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def ensure_canonical_weights() -> None:
    provenance = json.loads((ROOT / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))
    expected = {
        name: digest
        for name, digest in provenance["sha256"].items()
        if name.startswith("weights/")
    }
    weights_dir = ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    for relative, digest in expected.items():
        target = ROOT / relative
        if target.exists() and _sha256(target) == digest:
            continue

        url = f"{SOURCE}/{relative}"
        fd, tmp_name = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".download", dir=weights_dir
        )
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "TWM-Lab/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, tmp.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            actual = _sha256(tmp)
            if actual != digest:
                raise RuntimeError(
                    f"Downloaded checkpoint SHA-256 mismatch for {relative}: {actual}"
                )
            tmp.replace(target)
        finally:
            if tmp.exists():
                tmp.unlink()
