"""Score recorded verifier answers for an annotation manifest under all three configurations (offline, no model)."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.harness import evaluate, load_trials, result_json  # noqa: E402
from app.evaluation.manifest import load_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--include-draft", action="store_true", help="also score draft (unadjudicated) labels")
    parser.add_argument("--force-independent", action="store_true",
                        help="counterfactual: treat the verifier as independent of the generator")
    args = parser.parse_args()
    logging.disable(logging.ERROR)

    status = ("adjudicated", "draft") if args.include_draft else ("adjudicated",)
    result = evaluate(load_manifest(args.manifest), load_trials(args.trials), require_status=status,
                      force_independent=args.force_independent)
    print(result.result_label, file=sys.stderr)
    print(result_json(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
