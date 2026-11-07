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
from bifocalgemma.models.v2.initialization import describe_initialization
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
    native = sub.add_parser("smoke-native", help="Native full-depth tiny forward/backward.")
    native.add_argument("--experiment", choices=(*EXPERIMENT_NAMES, "all"), default="all")
    native.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train = sub.add_parser("train", help="Documented production placeholder.")
    train.add_argument("--experiment", choices=EXPERIMENT_NAMES, default=EXPERIMENT_NAMES[0])
    fast_train = sub.add_parser(
        "train-fast", help="Fast from-scratch training pipeline for BifocalGemma-2B."
    )
    fast_train.add_argument(
        "--model", choices=["bifocalgemma-2b", "baseline-2b"], default="bifocalgemma-2b"
    )
    fast_train.add_argument("--steps", type=int, default=2000)
    fast_train.add_argument("--output-dir", type=str, default="outputs/bifocalgemma_2b")
    fast_train.add_argument(
        "--data-dir", type=str, default=None, help="Directory containing training data"
    )
    fast_train.add_argument(
        "--pretrained-backbone-dir",
        type=str,
        default=None,
        help="Local directory containing pretrained backbone safetensors",
    )
    fast_train.add_argument(
        "--decoder-backbone",
        type=str,
        default="google/gemma-4-E2B-it",
        help="Hugging Face repo id or local path for decoder backbone",
    )
    fast_train.add_argument(
        "--encoder-backbone",
        type=str,
        default="google/gemma-4-E2B",
        help="Hugging Face repo id or local path for encoder backbone",
    )
    fast_train.add_argument(
        "--tokenizer-path",
        type=str,
        default="outputs/context_tokenizers/raw/google__gemma-4-E2B-it",
        help="Path to verified local tokenizer assets",
    )
    fast_train.add_argument("--push-to-hub", action="store_true")
    fast_train.add_argument("--hub-repo", type=str, default=None)
    fast_train.add_argument("--tiny", action="store_true")
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
            from bifocalgemma.models.v2.reference import run_reference_smoke

            selected = (
                EXPERIMENT_SPECS[:4]
                if args.experiment == "all"
                else (get_experiment(args.experiment),)
            )
            for spec in selected:
                print(json.dumps(run_reference_smoke(spec, args.device)))
        elif args.command == "smoke-native":
            import torch

            from bifocalgemma.models.v2.native import run_native_smoke

            torch.set_num_threads(min(4, torch.get_num_threads()))
            selected = (
                EXPERIMENT_SPECS if args.experiment == "all" else (get_experiment(args.experiment),)
            )
            for spec in selected:
                print(json.dumps(run_native_smoke(spec, args.device)))
        elif args.command == "train":
            from bifocalgemma.training.trainer import SFTConfig, train_direct_sft

            train_direct_sft(
                get_experiment(args.experiment),
                Path("data/manifest.jsonl"),
                SFTConfig(),
                Path("outputs"),
            )
        elif args.command == "train-fast":
            from bifocalgemma.training.fast_pipeline import FastTrainingConfig, FastTrainingPipeline

            if args.model == "baseline-2b":
                stage1 = 0
                stage2 = args.steps
            else:
                stage1 = args.steps // 2
                stage2 = args.steps - stage1
            cfg = FastTrainingConfig(
                model_choice=args.model,
                total_steps=args.steps,
                stage1_steps=stage1,
                stage2_steps=stage2,
                output_dir=args.output_dir,
                data_dir=args.data_dir,
                pretrained_backbone_dir=args.pretrained_backbone_dir,
                decoder_backbone=args.decoder_backbone,
                encoder_backbone=args.encoder_backbone,
                tokenizer_path=args.tokenizer_path,
                push_to_hub=args.push_to_hub,
                hub_repo=args.hub_repo,
                tiny=args.tiny,
            )
            pipeline = FastTrainingPipeline(cfg)
            summary = pipeline.run()
            print(json.dumps(summary, indent=2))
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
