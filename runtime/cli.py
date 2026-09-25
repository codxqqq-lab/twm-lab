"""Headless access to the very same inference callable as the Streamlit app."""

import argparse
import json
from pathlib import Path

from runtime.engine import verify_integrity
from runtime.schema import validate_scenario
from runtime.ui_adapter import predict_ui


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct frozen R13 neural inference")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), default=17)
    parser.add_argument("--actions", required=True, help="Comma-separated node indices, 1–16 steps")
    args = parser.parse_args()
    verify_integrity()
    scenario = validate_scenario(json.loads(args.scenario.read_text(encoding="utf-8")))
    actions = [int(a.strip()) for a in args.actions.split(",")]
    print(json.dumps(predict_ui(scenario["history"], actions, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
