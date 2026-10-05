"""Offline construction entrypoints. See docs/BUILD_GUIDE.md."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from bifocalgemma.config import (
    EXPERIMENT_NAMES,
    EXPERIMENT_SPECS,
    PROFILES,
    get_experiment,
    validate_experiment_matrix,
)
from bifocalgemma.models.initialization import describe_initialization
from bifocalgemma.scaffold import BUILD_BLOCKS, FeatureNotReadyError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BifocalGemma construction blocks; no automatic model downloads."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser(
        "inspect", help="Show checkpoint provenance and implementation status."
    )
    inspect.add_argument("--experiment", choices=EXPERIMENT_NAMES, default=EXPERIMENT_NAMES[0])
    validate = sub.add_parser("validate-config", help="Validate the checked-in six-model matrix.")
    validate.add_argument("--config", type=Path, default=Path("configs/experiments.json"))
    smoke = sub.add_parser("smoke", help="Tiny synthetic reference forward/backward only.")
    smoke.add_argument(
        "--experiment", choices=(*EXPERIMENT_NAMES[:4], "all"), default=EXPERIMENT_NAMES[0]
    )
    smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train = sub.add_parser("train", help="Documented production placeholder.")
    train.add_argument("--experiment", choices=EXPERIMENT_NAMES, default=EXPERIMENT_NAMES[0])
    evaluate = sub.add_parser("evaluate", help="Documented real-harness placeholder.")
    evaluate.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            spec = get_experiment(args.experiment)
            print(
                json.dumps(
                    {
                        "experiment": spec.to_dict(),
                        "native_profiles": {
                            "encoder": asdict(PROFILES[spec.encoder.size])
                            if spec.encoder
                            else None,
                            "decoder": asdict(PROFILES[spec.decoder.size]),
                        },
                        "initialization_plan": asdict(describe_initialization(spec)),
                        "blocks": [asdict(block) for block in BUILD_BLOCKS],
                        "pretrained_weights_loaded": False,
                    },
                    indent=2,
                )
            )
        elif args.command == "validate-config":
            specs = validate_experiment_matrix(args.config)
            print(json.dumps({"valid": True, "models": [spec.name for spec in specs]}))
        elif args.command == "smoke":
            from bifocalgemma.models.reference import run_reference_smoke

            selected = (
                EXPERIMENT_SPECS[:4]
                if args.experiment == "all"
                else (get_experiment(args.experiment),)
            )
            for spec in selected:
                print(json.dumps(run_reference_smoke(spec, args.device)))
        elif args.command == "train":
            from bifocalgemma.training.trainer import SFTConfig, train_direct_sft

            train_direct_sft(
                get_experiment(args.experiment),
                Path("data/manifest.jsonl"),
                SFTConfig(),
                Path("outputs"),
            )
        elif args.command == "evaluate":
            from bifocalgemma.evaluation.harness import SWEBenchHarness

            SWEBenchHarness().evaluate(args.predictions, "unselected", "unselected")
    except FeatureNotReadyError as error:
        parser.exit(2, f"{error}\n")
    except (ValueError, OSError) as error:
        parser.exit(2, f"Invalid input: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
