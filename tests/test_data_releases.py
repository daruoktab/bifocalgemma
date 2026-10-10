"""Release isolation, byte roundtrips and immutable HF consumer boundaries."""

import json
import zipfile

import pytest

from bifocalgemma.data.curation import export_pool
from bifocalgemma.data.offline_sft import stable_fingerprint
from bifocalgemma.data.releases import (
    archive_files,
    audit_split_disjointness,
    build_release,
    fetch_collection,
    publish_release,
    unpack_archive,
    verify_release,
)


def pool(tmp_path):
    records = []
    for split, index in (("train", 0), ("dev", 1)):
        records.append(
            {
                "instance_id": f"id-{index}",
                "pool_split": split,
                "role": "train",
                "instruction": "Change return value.",
                "before": "x = 0\n",
                "after": "x = 1\n",
                "path": "a.py",
                "target_patch": "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 0\n+x = 1\n",
                "source": "fixture",
                "source_revision": "a" * 40,
                "canonical_repos": [f"owner/repo{index}"],
                "commit": f"commit-{index}",
                "before_signature": f"before-{index}",
                "code_license": "mit",
            }
        )
    root = tmp_path / "pool"
    export_pool(
        records,
        root,
        {"tokens": {"limits": {"encoder": 8192, "pair_decoder": 2048, "baseline_decoder": 10752}}},
    )
    return root


def release(tmp_path):
    source = pool(tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "name": "test",
                "curator": "fixture",
                "date": "2026-10-10",
                "collections": [
                    {
                        "name": "foundation",
                        "kind": "curated_pool",
                        "directory": str(source),
                        "description": "Fixture only.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "release"
    build_release(spec, out)
    return source, out


def test_release_original_bytes_and_view_schema(tmp_path):
    import pyarrow.parquet as pq

    source, output = release(tmp_path)
    manifest = verify_release(output)
    item = manifest["collections"]["foundation"]
    assert item["counts"] == {"train": 1, "dev": 1}
    assert item["role"] == "train"
    view = pq.read_table(output / "collections/foundation/view/train.parquet").to_pylist()
    assert view[0]["target_patch"]
    destination = tmp_path / "restored"
    unpack_archive(
        output / item["archives"]["public"]["path"], item["archives"]["public"], destination
    )
    for name in ("train.jsonl", "dev.jsonl", "manifest.json"):
        assert (destination / name).read_bytes() == (source / name).read_bytes()


def test_release_rejects_corrupt_member(tmp_path):
    _, output = release(tmp_path)
    path = output / "collections/foundation/pool.zip"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="release_file_integrity"):
        verify_release(output)


def test_split_overlap_rejected():
    rows = {
        "train": [{"canonical_repos": ["a/b"], "commit": "x"}],
        "dev": [{"canonical_repos": ["a/b"], "commit": "y"}],
    }
    with pytest.raises(ValueError, match="lineage_overlap"):
        audit_split_disjointness(rows)


def test_archive_source_escape_rejected(tmp_path):
    (tmp_path / "root").mkdir()
    (tmp_path / "secret").write_text("outside", encoding="utf-8")
    with pytest.raises(ValueError, match="path"):
        archive_files(tmp_path / "root", ["../secret"], tmp_path / "data.zip")


def test_eval_public_and_gold_separate(tmp_path):
    root = tmp_path / "eval"
    (root / "eval_private").mkdir(parents=True)
    (root / "eval_inputs.jsonl").write_text(
        json.dumps({"instance_id": "e1", "role": "eval", "context": "x = 0", "instruction": "Fix."})
        + "\n"
    )
    (root / "eval_private/specs.jsonl").write_text(
        json.dumps({"instance_id": "e1", "gold_patch": "secret target"}) + "\n"
    )
    (root / "preparation_report.json").write_text('{"accepted": 1}')
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "name": "test",
                "curator": "fixture",
                "date": "2026-10-10",
                "collections": [
                    {
                        "name": "benchmark",
                        "kind": "prepared_code_eval",
                        "directory": str(root),
                        "description": "Fixture only.",
                    }
                ],
            }
        )
    )
    output = tmp_path / "out"
    manifest = build_release(spec, output)
    archives = manifest["collections"]["benchmark"]["archives"]
    with zipfile.ZipFile(output / archives["public"]["path"]) as z:
        assert "eval_private/specs.jsonl" not in z.namelist()
    with zipfile.ZipFile(output / archives["gold"]["path"]) as z:
        assert z.namelist() == ["eval_private/specs.jsonl"]
    import pyarrow.parquet as pq

    assert "gold_patch" not in pq.read_schema(output / "collections/benchmark/view/test.parquet")


def test_moving_revision_rejected_without_network(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        fetch_collection("owner/repo", "main", "training", tmp_path / "out")


def test_train_fetch_never_downloads_eval_gold(monkeypatch, tmp_path):
    import huggingface_hub

    release = {
        "format": "bifocalgemma_dataset_release_v1",
        "collections": {"eval": {"role": "eval", "archives": {"gold": {"path": "gold.zip"}}}},
    }
    release["fingerprint"] = stable_fingerprint(release)
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps(release))
    downloads = []

    def download(repo, name, **kwargs):
        downloads.append(name)
        return str(manifest)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(ValueError, match="role_mismatch"):
        fetch_collection("owner/repo", "a" * 40, "eval", tmp_path / "out")
    assert downloads == ["release.json"]


def test_fetch_pinned_pool_roundtrip_and_reuse(monkeypatch, tmp_path):
    import huggingface_hub

    source, output = release(tmp_path)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda repo, name, **kwargs: str(output / name)
    )
    destination = tmp_path / "download"
    result = fetch_collection("owner/data", "a" * 40, "foundation", destination)
    assert result["fingerprint"]
    assert (source / "train.jsonl").read_bytes() == (destination / "train.jsonl").read_bytes()
    assert fetch_collection("owner/data", "a" * 40, "foundation", destination) == result


def test_publish_refuses_public_repo_before_upload(monkeypatch, tmp_path):
    import huggingface_hub

    _, output = release(tmp_path)

    class Api:
        def whoami(self):
            return {"name": "owner"}

        def repo_exists(self, *args, **kwargs):
            return True

        def dataset_info(self, *args, **kwargs):
            class Info:
                private = False

            return Info()

        def upload_folder(self, **kwargs):
            raise AssertionError("Must not upload")

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    with pytest.raises(ValueError, match="public repo"):
        publish_release(output, "owner/data")
