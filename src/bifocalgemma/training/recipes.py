"""Shared experimental recipes and training-length-relative I/O cadence."""

import copy
import math
from pathlib import PurePosixPath

from bifocalgemma.training.policy import default_training_profiles, validate_training_policy


def comparison_recipes(family):
    """Same data/LoRA/objective/LR numbers in every architecture and recipe.

    These are conservative experimental defaults, not a claim of an optimum.
    The combined optimized arm cannot isolate each optimizer mechanism.
    """
    policy = default_training_profiles()[family]
    policy["alignment_fp32"] = True
    for settings in policy["groups"].values():
        settings.update(
            lr=2e-5, warmup_steps=0, warmup_ratio=0.03, scheduler="cosine", min_lr_ratio=0.1
        )
    baseline = copy.deepcopy(policy)
    baseline["optimizer"] = {
        "name": "adamw",
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    optimized = copy.deepcopy(policy)
    optimized["optimizer"] = {
        "name": "grok_orscale_lora_v1",
        "weight_decay": 0.0,
        "grok_decay": 0.98,
        "grok_amplification": 0.5,
        "momentum": 0.95,
        "ns_steps": 5,
        "trust_min": 0.5,
        "trust_max": 2.0,
    }
    return {"base": baseline, "optimized": optimized}


def resolve_training_cadence(
    total_steps, *, eval_every=None, save_every=None, eval_fraction=0.25, save_fraction=0.10
):
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("Training horizon must be a positive integer.")
    result = {"total_steps": total_steps}
    for name, explicit, fraction in (
        ("eval_every", eval_every, eval_fraction),
        ("save_every", save_every, save_fraction),
    ):
        if (
            not isinstance(fraction, (int, float))
            or not math.isfinite(fraction)
            or not 0 < fraction <= 1
        ):
            raise ValueError("Cadence fraction must be in (0, 1].")
        if explicit is not None and (type(explicit) is not int or explicit < 1):
            raise ValueError("Explicit cadence must be a positive integer.")
        result[name] = (
            explicit if explicit is not None else max(1, math.ceil(total_steps * fraction))
        )
    return result


def run_recipe_comparison(*, recipes, experiments, run_name, resume_from=None, **settings):
    """Train/upload every model/recipe before evaluating any, one resident model at a time."""
    from bifocalgemma.training.sequential import run_sequential_pipeline

    if not recipes or set(recipes) - {"base", "optimized"}:
        raise ValueError("Choose base and/or optimized recipes.")
    run_path = PurePosixPath(run_name)
    if (
        run_path.is_absolute()
        or len(run_path.parts) != 1
        or run_path.name in {".", ".."}
        or "\\" in run_name
    ):
        raise ValueError("Comparison RUN_NAME must be a single path component.")
    if not experiments or len(set(experiments)) != len(experiments):
        raise ValueError("Choose unique experiments.")
    if resume_from and len(recipes) * len(experiments) != 1:
        raise ValueError(
            "Use AUTO_RESUME for a comparison; explicit resume needs one model/recipe."
        )
    # Validate every recipe before launching the first GPU load.
    for policy in recipes.values():
        validate_training_policy(policy)
        if policy["family"] != settings["family"]:
            raise ValueError("Recipe family differs from notebook family.")
    evaluate = settings.pop("run_benchmarks", True)
    train = settings.pop("run_training", True)
    jobs = [
        (experiment, recipe, policy)
        for experiment in experiments
        for recipe, policy in recipes.items()
    ]
    results = []
    # The first pass is a barrier: no benchmark downloads/inference until every
    # training run and its requested training upload has returned successfully.
    for phase in ["training", "benchmark"] if evaluate else ["training"]:
        phase_results = []
        print(f"Comparison phase: {phase}; {len(jobs)} model/recipe runs", flush=True)
        phase_settings = dict(settings)
        if phase == "benchmark":
            # Reuse pass-one checkpoints even when the initial request was a fresh run.
            phase_settings["auto_resume"] = True
        for job_index, (experiment, recipe, policy) in enumerate(jobs, 1):
            print(f"{phase} {job_index}/{len(jobs)}: {experiment}/{recipe}", flush=True)
            outcome = run_sequential_pipeline(
                experiments=(experiment,),
                run_name=f"{run_name}/{recipe}",
                training_policy=copy.deepcopy(policy),
                resume_from=resume_from,
                recipe_name=recipe,
                run_training=train if phase == "training" else False,
                run_benchmarks=phase == "benchmark",
                finalize=phase == "benchmark" or not evaluate,
                **phase_settings,
            )
            for row in outcome["models"]:
                row["recipe"] = recipe
                phase_results.append(row)
        results = phase_results
    return {
        "status": "PENDING_UPLOAD"
        if any(r["status"] == "PENDING_UPLOAD" for r in results)
        else "COMPLETED",
        "family": settings["family"],
        "sequence": list(experiments),
        "recipes": list(recipes),
        "models": results,
    }
