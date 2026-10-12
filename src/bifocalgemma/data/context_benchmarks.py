"""Pinned static context benchmarks; gold is exported separately from model inputs."""

import hashlib
import json
import tarfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from bifocalgemma.data.offline_sft import stable_fingerprint
from bifocalgemma.data.repository import read_source_rows
from bifocalgemma.training.stages import atomic_json

SWEQA_REVISION = "579a8e1220c46742213998a36ccbb26c8bd26468"
REPOFORMER_REVISION = "5b0571318e9918fd2af132c4b35077f9ea331133"
LONGCODE_REVISION = "989d5eff750d65a72c522e47f8f745ef2e22906b"


def fetch_public(url, path, *, maximum_bytes=200_000_000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    temporary = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "BifocalGemma-context-audit"})
    with urllib.request.urlopen(request, timeout=45) as response, temporary.open("wb") as stream:
        total = 0
        while data := response.read(1024 * 1024):
            total += len(data)
            if total > maximum_bytes:
                raise ValueError("source_download_budget")
            stream.write(data)
    temporary.replace(path)
    return path


def write_context_eval(directory, public, gold, protocol):
    directory = Path(directory)
    ids = [row["instance_id"] for row in public]
    if not ids or len(set(ids)) != len(ids) or ids != [row["instance_id"] for row in gold]:
        raise ValueError("context_benchmark_identity")
    for row in public:
        if row.get("role") != "eval" or any(
            k in row
            for k in (
                "answer",
                "correct_answer",
                "correct_letter",
                "groundtruth",
                "reference",
                "is_hard",
            )
        ):
            raise ValueError("context_benchmark_gold_in_input")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "eval_private").mkdir()
    for path, rows in (
        (directory / "eval_inputs.jsonl", public),
        (directory / "eval_private/specs.jsonl", gold),
    ):
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "format": "bifocalgemma_context_eval_v1",
        "count": len(public),
        "protocol": protocol,
        "official_protocol": False,
        "rejections": [],
        "repositories": sorted({r["repos"] for r in public}),
        "public_sha256": hashlib.sha256((directory / "eval_inputs.jsonl").read_bytes()).hexdigest(),
        "gold_sha256": hashlib.sha256(
            (directory / "eval_private/specs.jsonl").read_bytes()
        ).hexdigest(),
    }
    report["fingerprint"] = stable_fingerprint(report)
    atomic_json(directory / "preparation_report.json", report)
    return report


def prepare_sweqa(cache, output):
    from huggingface_hub import hf_hub_download

    reports = {}
    identities = []
    aliases = {"pylint": "pylint-dev/pylint", "pytest": "pytest-dev/pytest"}
    for condition in ("oracle", "noisy_oracle"):
        source = hf_hub_download(
            "lailaelkoussy/swe-qa",
            f"data/{condition}-00000-of-00001.parquet",
            repo_type="dataset",
            revision=SWEQA_REVISION,
            cache_dir=cache,
        )
        public, gold = [], []
        seen = set()
        for row in read_source_rows(source):
            key = stable_fingerprint([row["question"], row["options"], row["correct_answer"]])
            if key in seen:
                raise ValueError("duplicate_sweqa_question")
            seen.add(key)
            options = row["options"]
            question = row["question"] + "\n" + "\n".join(f"{k}) {options[k]}" for k in "ABCD")
            instruction = (
                "Answer the code comprehension question. Return only A, B, C, or D.\n" + question
            )
            repo = aliases.get(row["repo"], row["repo"]).lower()
            paths = sorted({chunk["path"] for chunk in row["chunks"] if chunk.get("path")})
            public.append(
                {
                    "instance_id": "sweqa-" + key,
                    "role": "eval",
                    "task_type": "mcq",
                    "benchmark": "SWE-QA",
                    "source": "lailaelkoussy/swe-qa",
                    "source_revision": SWEQA_REVISION,
                    "repos": repo,
                    "condition": condition,
                    "category": row["category"],
                    "instruction": instruction,
                    "context": instruction + "\n\n" + row["code"],
                    "visible_paths": paths,
                    "task_scope": "conditional_code_comprehension",
                    "max_encoder_tokens": 8192,
                    "max_new_tokens": 32,
                    "context_policy": "released_" + condition,
                    "official_protocol": False,
                    "dataset_license": "apache-2.0",
                }
            )
            gold.append(
                {"instance_id": "sweqa-" + key, "answer": row["correct_answer"], "task_type": "mcq"}
            )
        public.sort(key=lambda r: r["instance_id"])
        by_id = {r["instance_id"]: r for r in gold}
        gold = [by_id[r["instance_id"]] for r in public]
        identities.append(set(by_id))
        reports[condition] = write_context_eval(
            Path(output) / ("sweqa_" + condition + "_v1"),
            public,
            gold,
            {
                "task": "mcq",
                "condition": condition,
                "scorer": "strict_letter_accuracy_v1",
                "oracle_context": True,
                "unique_questions_across_conditions": 9072,
            },
        )
    if identities[0] != identities[1] or len(identities[0]) != 9072:
        raise ValueError("sweqa_condition_cohort_changed")
    return reports


def prepare_crosscodelong(cache, output):
    reports = {}
    for kind in ("chunk", "function"):
        url = (
            f"https://raw.githubusercontent.com/amazon-science/Repoformer/{REPOFORMER_REVISION}/"
            f"crosscodelongeval/cceval_{kind}_eval_data.tar.gz"
        )
        source = fetch_public(url, Path(cache) / f"crosscode_{kind}.tar.gz")
        records = []
        with tarfile.open(source, "r:gz") as archive:
            names = [
                m
                for m in archive
                if m.isfile()
                and not Path(m.name).name.startswith("._")
                and m.name.endswith("_sparse_rg1.jsonl")
            ]
            if len(names) != 1:
                raise ValueError("crosscode_expected_target_free_retrieval_file")
            stream = archive.extractfile(names[0])
            assert stream is not None
            records = [json.loads(line) for line in stream if line.strip()]
        public, gold = [], []
        for row in records:
            metadata = row["metadata"]
            task_id = metadata["task_id"]
            if "querywithoutgt" not in task_id:
                raise ValueError("crosscode_gold_in_retrieval_query")
            identifier = "ccl-" + stable_fingerprint([kind, task_id])
            repository = metadata["filename"].split("/")[0]
            cross = row["crossfile_context"]
            instruction = (
                "Complete the missing Python code between PREFIX and SUFFIX. Return only that code."
            )
            # Preserve the released retrieval material as text, without injecting the target.
            cross_text = cross if isinstance(cross, str) else json.dumps(cross, ensure_ascii=False)
            context = (
                instruction
                + "\n\n### CROSS-FILE CONTEXT\n"
                + cross_text
                + "\n\n### PREFIX\n"
                + row["prompt"]
                + "\n### MISSING CODE\n"
                + "\n### SUFFIX\n"
                + row["right_context"]
            )
            public.append(
                {
                    "instance_id": identifier,
                    "role": "eval",
                    "task_type": "completion",
                    "benchmark": "CrossCodeLongEval",
                    "source": "amazon-science/Repoformer",
                    "source_revision": REPOFORMER_REVISION,
                    "repos": repository,
                    "repository_slug": repository,
                    "upstream_task_id": task_id,
                    "instruction": instruction,
                    "context": context,
                    "completion_kind": kind,
                    "task_scope": "cross_file_completion",
                    "context_policy": "released_sparse_rg1_fim",
                    "max_encoder_tokens": 8192,
                    "max_new_tokens": 1024,
                    "official_protocol": False,
                    "dataset_license": "apache-2.0",
                }
            )
            gold.append(
                {
                    "instance_id": identifier,
                    "reference": row["groundtruth"],
                    "task_type": "completion",
                }
            )
        if len(public) != 5000:
            raise ValueError("crosscode_source_count_changed")
        reports[kind] = write_context_eval(
            Path(output) / ("crosscodelong_" + kind + "_source_v1"),
            public,
            gold,
            {
                "task": "completion",
                "variant": "left_right_crossfile_rg1",
                "oracle": False,
                "scorer": "adapted_whitespace_em_sequence_similarity_regex_identifier_set_f1_v2",
            },
        )
    return reports


def prepare_longcodeqa(cache, output):
    from huggingface_hub import hf_hub_download

    source = hf_hub_download(
        "Steefano/LCB",
        "LongCodeQA.zip",
        repo_type="dataset",
        revision=LONGCODE_REVISION,
        cache_dir=cache,
    )
    with zipfile.ZipFile(source) as archive:
        records = json.loads(archive.read("LQA/32K.json"))
        all_repos = set()
        for name in archive.namelist():
            if name.startswith("LQA/") and name.endswith(".json"):
                all_repos.update(r["repo"].lower() for r in json.loads(archive.read(name)))
    public, gold = [], []
    for row in records:
        identifier = "lcqa-" + stable_fingerprint([row["repo"], row["question"]])
        instruction = row["prompt_goal"] + "\n" + row["question"]
        public.append(
            {
                "instance_id": identifier,
                "role": "eval",
                "task_type": "mcq",
                "benchmark": "LongCodeQA",
                "source": "Steefano/LCB",
                "source_revision": LONGCODE_REVISION,
                "repos": row["repo"].lower(),
                "instruction": instruction,
                "context": row["prompt"],
                "task_scope": "repository_comprehension",
                "max_encoder_tokens": 32768,
                "max_new_tokens": 32,
                "context_policy": "full_released_32k_prompt",
                "official_protocol": False,
                "dataset_license": "apache-2.0",
            }
        )
        gold.append(
            {"instance_id": identifier, "answer": row["correct_letter"], "task_type": "mcq"}
        )
    if len(public) != 113:
        raise ValueError("longcodeqa_source_count_changed")
    report = write_context_eval(
        Path(output) / "longcodeqa_32k_source_v1",
        public,
        gold,
        {
            "task": "mcq",
            "upstream_context_bucket": "32K",
            "scorer": "strict_letter_accuracy_v1",
            "all_upstream_repositories_quarantined": sorted(all_repos),
        },
    )
    return report


def matched_eval_cohort(source, output, tokenizer_cache, *, encoder_limit, decoder_limit=8192):
    """Freeze identical IDs for six models before seeing predictions; never crop context."""
    from bifocalgemma.config import EXPERIMENT_SPECS
    from bifocalgemma.data.tokenization import load_verified_tokenizer
    from bifocalgemma.evaluation.context import prompt_ids

    assets = {}
    asset_signatures = {}
    for spec in EXPERIMENT_SPECS:
        for ref in (spec.encoder, spec.decoder):
            if ref and ref.model_id not in assets:
                assets[ref.model_id], manifest = load_verified_tokenizer(ref, Path(tokenizer_cache))
                asset_signatures[ref.model_id] = stable_fingerprint(manifest["files"])
    public = list(read_source_rows(Path(source) / "eval_inputs.jsonl"))
    gold = {
        r["instance_id"]: r for r in read_source_rows(Path(source) / "eval_private/specs.jsonl")
    }
    kept, references, excluded = [], [], []
    for row in public:
        try:
            lengths = {}
            token_cache = {}
            for spec in EXPERIMENT_SPECS:
                token_key = (
                    asset_signatures[spec.encoder.model_id] if spec.encoder else None,
                    asset_signatures[spec.decoder.model_id],
                )
                if token_key not in token_cache:
                    token_cache[token_key] = prompt_ids(
                        row,
                        spec,
                        assets.get(spec.encoder.model_id) if spec.encoder else None,
                        assets[spec.decoder.model_id],
                        encoder_limit=encoder_limit,
                    )
                enc, prefix = token_cache[token_key]
                lengths[spec.name] = {"encoder": len(enc), "prefix": len(prefix)}
            paired_prefix = max(
                lengths[spec.name]["prefix"] for spec in EXPERIMENT_SPECS if spec.encoder
            )
            generation_budget = min(
                32 if row["task_type"] == "mcq" else 8192, decoder_limit - paired_prefix
            )
            if generation_budget < 1:
                raise ValueError("decoder_prefix_exceeds_budget")
            if row.get("task_type") == "completion":
                reference = gold[row["instance_id"]]["reference"]
                if len(reference.strip()) >= 64 and reference in row["context"]:
                    raise ValueError("reference_already_visible_in_context")
                if any(
                    len(assets[spec.decoder.model_id].encode(reference, add_special_tokens=False))
                    > generation_budget
                    for spec in EXPERIMENT_SPECS
                ):
                    raise ValueError("reference_exceeds_fixed_generation_budget")
            for spec in EXPERIMENT_SPECS:
                cap = decoder_limit if spec.encoder else encoder_limit + decoder_limit + 512
                if lengths[spec.name]["prefix"] + generation_budget > cap:
                    raise ValueError("decoder_generation_budget")
            kept.append(
                {
                    **row,
                    "max_encoder_tokens": encoder_limit,
                    "max_new_tokens": generation_budget,
                    "token_lengths": lengths,
                }
            )
            references.append(gold[row["instance_id"]])
        except ValueError as exc:
            excluded.append({"instance_id": row["instance_id"], "reason": str(exc)})
    source_report = json.loads(
        (Path(source) / "preparation_report.json").read_text(encoding="utf-8")
    )
    protocol = {
        **source_report["protocol"],
        "matched_six_model_cohort": True,
        "max_encoder_tokens": encoder_limit,
        "max_paired_decoder_tokens": decoder_limit,
        "source_count": len(public),
        "eligible_count": len(kept),
        "silent_truncation": False,
    }
    report = write_context_eval(output, kept, references, protocol)
    report["rejections"] = excluded
    report["exclusion_counts"] = dict(Counter(r["reason"] for r in excluded))
    report["fingerprint"] = stable_fingerprint(
        {k: v for k, v in report.items() if k != "fingerprint"}
    )
    atomic_json(Path(output) / "preparation_report.json", report)
    return report
