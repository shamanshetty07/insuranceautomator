"""Opt-in smoke test: real local Qwen-VL inference on images you choose. Not run by pytest.

Needs `pip install -e ".[vlm]"` and an already-downloaded local model directory; it
never downloads anything (offline mode is forced). No database is used.

  VLM_SMOKE=1 python scripts/vlm_smoke.py --model-dir ../models/Qwen2-VL-2B-Instruct \\
      [--sha256 <pinned digest>] [--device cpu|mps] img1.jpg [img2.jpg ...]

Prints the model digest, timings, the validated structured claims, and the verifier's
label for each claim. Output is a model prediction, not ground truth.
"""

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.types import EvidenceBundle, ImageRef, ReportClaim  # noqa: E402
from app.services.vlm.adapters import QwenClaimVerifier, QwenStructuredReportGenerator, VlmLimits  # noqa: E402
from app.services.vlm.errors import VlmError  # noqa: E402
from app.services.vlm.transformers_runner import TransformersQwenRunner  # noqa: E402


def main() -> int:
    if os.environ.get("VLM_SMOKE") != "1":
        print("Refusing to run: set VLM_SMOKE=1 to load a real model.", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--name", default="qwen2-vl-2b-instruct")
    parser.add_argument("--version", default="local")
    parser.add_argument("--sha256", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()

    started = time.monotonic()
    runner = TransformersQwenRunner(
        args.model_dir, args.name, args.version, device=args.device, dtype=args.dtype, expected_sha256=args.sha256
    )
    print(f"loaded {runner.info} in {time.monotonic() - started:.1f}s; runtime={dict(runner.runtime)}")
    limits = VlmLimits(timeout_seconds=args.timeout)
    evidence = EvidenceBundle(images=[ImageRef(image_id=uuid.uuid4(), path=p) for p in args.images])

    try:
        started = time.monotonic()
        report = QwenStructuredReportGenerator(runner, limits).generate_structured(evidence)
        print(f"\ngenerated in {time.monotonic() - started:.1f}s:\n{report.model_dump_json(indent=2)}")
        verifier = QwenClaimVerifier(runner, limits)
        for claim in report.claims.claims:
            images = EvidenceBundle(images=[r for r in evidence.images if r.image_id in claim.image_ids])
            if not images.images:
                print(f"{claim.claim_id}: cites no image; not verified")
                continue
            started = time.monotonic()
            (verdict,) = verifier.verify([ReportClaim(claim.text)], images)
            print(f"{claim.claim_id}: {verdict.label} ({time.monotonic() - started:.1f}s) {verdict.explanation}")
    except VlmError as exc:
        print(f"VLM error [{exc.code}]: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
