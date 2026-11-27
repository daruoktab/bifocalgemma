"""Execute the generated standalone loader, including the Molab missing-ZIP case."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools import generate_full_matrix_notebooks as generator

REVISION = "a" * 40


def archive_bytes(name="src/fixture_package/__init__.py"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(name, "VALUE = 42\n")
    return stream.getvalue()


def loader(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    rendered = generator.render_notebook("2base-2it", {})
    tree = ast.parse(rendered)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load_runtime"
    )
    namespace: dict[str, Any] = {}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "generated_runtime_loader.py", "exec"
        ),
        namespace,
    )
    return namespace["load_runtime"]


def settings(payload, revision=REVISION):
    digest = hashlib.sha256(payload).hexdigest()
    return dict(
        repo_id="example/runtime",
        filename=f"runtime/runtime-{digest[:16]}.zip",
        revision=revision,
        expected_sha=digest,
    )


def no_network(*args, **kwargs):
    raise AssertionError("No download should be attempted.")


def test_local_preview_missing_archive_explains_publication_without_network(monkeypatch, tmp_path):
    load = loader(monkeypatch, tmp_path)
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    with pytest.raises(ValueError, match="unpublished LOCAL-PREVIEW"):
        load(
            **settings(archive_bytes(), revision="LOCAL-PREVIEW"),
            notebook_file="/tmp/marimo_167/__marimo__cell_MJUe_.py",
        )
    assert not (tmp_path / "outputs").exists()


def test_verified_local_preview_activates_archive_without_workspace_override(monkeypatch, tmp_path):
    load = loader(monkeypatch, tmp_path)
    payload = archive_bytes()
    kwargs = settings(payload, revision="LOCAL-PREVIEW")
    notebook = tmp_path / "repo/notebooks/train.py"
    local = notebook.parent / "runtime" / Path(kwargs["filename"]).name
    local.parent.mkdir(parents=True)
    local.write_bytes(payload)
    workspace_source = notebook.parent.parent / "src"
    workspace_source.mkdir()
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    root = load(**kwargs, notebook_file=notebook)
    assert (root / "src/fixture_package/__init__.py").read_text() == "VALUE = 42\n"
    assert sys.path[:2] == [str(root), str(root / "src")]
    assert str(workspace_source) not in sys.path


def test_corrupt_local_archive_uses_verified_cache(monkeypatch, tmp_path):
    load = loader(monkeypatch, tmp_path)
    payload = archive_bytes()
    kwargs = settings(payload)
    notebook = tmp_path / "notebooks/train.py"
    local = notebook.parent / "runtime" / Path(kwargs["filename"]).name
    local.parent.mkdir(parents=True)
    local.write_bytes(b"incomplete local archive")
    cached = tmp_path / "outputs/v3/runtime" / kwargs["expected_sha"][:16] / "runtime.zip"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(payload)
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    root = load(**kwargs, notebook_file=notebook)
    assert root == cached.parent
    assert cached.read_bytes() == payload


@pytest.mark.parametrize("first_url_fails", [False, True])
def test_standalone_molab_cell_downloads_pinned_verified_runtime(
    monkeypatch, tmp_path, first_url_fails
):
    load = loader(monkeypatch, tmp_path)
    payload = archive_bytes()
    kwargs = settings(payload)
    urls = []

    def download(request, **options):
        urls.append(request.full_url)
        if first_url_fails and len(urls) == 1:
            raise OSError("model repository unavailable")
        return io.BytesIO(payload)

    monkeypatch.setattr(urllib.request, "urlopen", download)
    root = load(**kwargs, notebook_file="/tmp/marimo_167/__marimo__cell_MJUe_.py")
    assert len(urls) == (2 if first_url_fails else 1)
    assert all(f"/resolve/{REVISION}/{kwargs['filename']}" in url for url in urls)
    assert (root / "runtime.zip").read_bytes() == payload


def test_bad_download_never_activates_code(monkeypatch, tmp_path):
    load = loader(monkeypatch, tmp_path)
    paths_before = list(sys.path)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"wrong runtime"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load(**settings(archive_bytes()))
    assert sys.path == paths_before
    assert not (tmp_path / "outputs").exists()


def test_unsafe_archive_path_is_rejected_after_checksum_verification(monkeypatch, tmp_path):
    load = loader(monkeypatch, tmp_path)
    payload = archive_bytes("../../../../escape.py")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    paths_before = list(sys.path)
    with pytest.raises(ValueError, match="unsafe path"):
        load(**settings(payload))
    assert sys.path == paths_before
    assert not (tmp_path / "escape.py").exists()


def test_release_check_rejects_local_preview(monkeypatch, tmp_path):
    payload = archive_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    manifest = tmp_path / "notebooks/runtime-manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"revision": "LOCAL-PREVIEW", "sha256": digest}))
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    monkeypatch.setattr(generator, "matrix_bundle", lambda root: (payload, digest))
    monkeypatch.setattr(sys, "argv", ["generator", "--check", "--require-published"])
    with pytest.raises(SystemExit, match="not published"):
        generator.main()


@pytest.mark.parametrize("arguments", [["--publish", "--check"], ["--require-published"]])
def test_invalid_generation_modes_cannot_publish(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["generator", *arguments])
    with pytest.raises(SystemExit) as error:
        generator.main()
    assert error.value.code == 2


@pytest.mark.parametrize("corrupt_download", [False, True])
def test_publication_only_adds_verified_immutable_runtime(monkeypatch, tmp_path, corrupt_download):
    payload = archive_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    filename = f"runtime/runtime-{digest[:16]}.zip"
    notebooks = tmp_path / "notebooks"
    runtime_dir = notebooks / "runtime"
    runtime_dir.mkdir(parents=True)
    manifest_path = notebooks / "runtime-manifest.json"
    previous_manifest = b'{"revision": "previous"}\n'
    manifest_path.write_bytes(previous_manifest)
    notebook_path = notebooks / "train_2base_2it.py"
    previous_notebook = b"# Previous notebook\n"
    notebook_path.write_bytes(previous_notebook)
    previous_runtime = runtime_dir / "runtime-previous.zip"
    previous_runtime.write_bytes(b"previous runtime")
    latest_runtime = runtime_dir / "latest.zip"
    latest_runtime.write_bytes(b"legacy latest runtime")
    downloaded = tmp_path / "downloaded.zip"
    downloaded.write_bytes(b"corrupt runtime" if corrupt_download else payload)
    uploads: list[dict[str, Any]] = []
    downloads: list[dict[str, Any]] = []

    class Api:
        def upload_file(self, **kwargs):
            uploads.append(kwargs)
            return SimpleNamespace(oid=REVISION)

    def download(repo_id, path, **kwargs):
        downloads.append({"repo_id": repo_id, "filename": path, **kwargs})
        # The currently usable notebook stays intact until the remote bytes verify.
        assert manifest_path.read_bytes() == previous_manifest
        assert notebook_path.read_bytes() == previous_notebook
        return str(downloaded)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=Api,
            hf_hub_download=download,
        ),
    )
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    monkeypatch.setattr(generator, "matrix_bundle", lambda root: (payload, digest))
    monkeypatch.setattr(
        generator,
        "MATRIX_NOTEBOOK_SPECS",
        {
            "2base-2it": generator.MATRIX_NOTEBOOK_SPECS["2base-2it"],
        },
    )
    monkeypatch.setattr(generator, "format_code", lambda code, filename: code)
    monkeypatch.setattr(sys, "argv", ["generator", "--publish"])

    if corrupt_download:
        with pytest.raises(ValueError, match="Published runtime checksum mismatch"):
            generator.main()
        assert manifest_path.read_bytes() == previous_manifest
        assert notebook_path.read_bytes() == previous_notebook
        assert not (runtime_dir / Path(filename).name).exists()
    else:
        generator.main()
        manifest = {
            "repo_id": generator.CENTRALIZED_HF_REPO,
            "filename": filename,
            "revision": REVISION,
            "sha256": digest,
        }
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
        assert notebook_path.read_text(encoding="utf-8") == generator.render_notebook(
            "2base-2it", manifest
        )
        assert (runtime_dir / Path(filename).name).read_bytes() == payload

    assert len(uploads) == 1
    assert uploads[0]["repo_id"] == generator.CENTRALIZED_HF_REPO
    assert uploads[0]["path_in_repo"] == filename
    assert uploads[0]["path_or_fileobj"] == payload
    assert downloads == [
        {
            "repo_id": generator.CENTRALIZED_HF_REPO,
            "filename": filename,
            "revision": REVISION,
            "token": False,
            "force_download": True,
        }
    ]
    assert previous_runtime.read_bytes() == b"previous runtime"
    assert latest_runtime.read_bytes() == b"legacy latest runtime"


def test_checked_in_matrix_notebooks_require_published_runtime():
    for key in generator.MATRIX_NOTEBOOK_SPECS:
        path = generator.ROOT / "notebooks" / f"train_{key.replace('-', '_')}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "runtime_manifest"
                for target in node.targets
            )
        )
        revision = ast.literal_eval(assignment.value)["revision"]
        assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision), (
            f"{path.name} cannot ship a LOCAL-PREVIEW runtime to Molab; publish before push."
        )


def test_notebook_launcher_handles_bifocalgemma_workspace_name_and_tools_provenance(
    tmp_path, monkeypatch
):
    from types import ModuleType, SimpleNamespace

    from tools import notebook_launcher

    workspace = tmp_path / "bifocalgemma"
    runtime_root = workspace / "outputs/runtime/abc"
    for base, text in (
        (workspace, b"VALUE = 1\n"),
        (runtime_root, b"VALUE = 1\r\n"),
    ):
        (base / "src/bifocalgemma/training").mkdir(parents=True)
        (base / "tools").mkdir(parents=True, exist_ok=True)
        (base / "src/bifocalgemma/__init__.py").write_bytes(b"")
        (base / "src/bifocalgemma/training/__init__.py").write_bytes(b"")
        (base / "src/bifocalgemma/training/matrix.py").write_bytes(text)
        (base / "tools/__init__.py").write_bytes(b"")
        (base / "tools/molab_matrix_support.py").write_bytes(text)

    # Double "bifocalgemma" in workspace path must still map to runtime_root/src/bifocalgemma/...
    loaded_pkg_file = workspace / "src/bifocalgemma/training/matrix.py"
    expected = notebook_launcher._expected_bifocalgemma_path(runtime_root, loaded_pkg_file)
    assert expected == runtime_root / "src/bifocalgemma/training/matrix.py"

    mod_pkg = ModuleType("bifocalgemma.training.matrix")
    mod_pkg.__file__ = str(loaded_pkg_file)
    mod_tools = ModuleType("tools.molab_matrix_support")
    mod_tools.__file__ = str(workspace / "tools/molab_matrix_support.py")
    fake_modules: dict[str, Any] = {
        "bifocalgemma.training.matrix": mod_pkg,
        "tools.molab_matrix_support": mod_tools,
    }
    fake_sys = SimpleNamespace(
        modules=fake_modules,
        path=list(sys.path),
        platform=sys.platform,
    )
    monkeypatch.setattr(notebook_launcher, "sys", fake_sys)
    notebook_launcher._validate_loaded_modules(runtime_root)

    # Full prepare_runtime runs mocked dependency check (never mutating conda) and validates provenance
    monkeypatch.setattr(
        notebook_launcher.runpy,
        "run_path",
        lambda path: {
            "bootstrap_notebook_dependencies": lambda pins, reinstall=False: {"status": "READY"}
        },
    )
    prep_result = notebook_launcher.prepare_runtime(runtime_root)
    assert prep_result == {"status": "READY"}
    assert fake_sys.path[:2] == [str(runtime_root.resolve()), str((runtime_root / "src").resolve())]

    # Stale tools.* module with differing bytes must trigger restart error
    (workspace / "tools/molab_matrix_support.py").write_text("VALUE = 999\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Restart kernel"):
        notebook_launcher.prepare_runtime(runtime_root)

    # Stale bundle-only tools.* module absent from target runtime must trigger restart error
    other_bundle = tmp_path / "other_runtime"
    (other_bundle / "tools").mkdir(parents=True)
    stale_only = other_bundle / "tools/stale_helper.py"
    stale_only.write_text("X = 1\n", encoding="utf-8")
    fake_modules.pop("tools.molab_matrix_support")
    mod_stale = ModuleType("tools.stale_helper")
    mod_stale.__file__ = str(stale_only)
    fake_modules["tools.stale_helper"] = mod_stale
    with pytest.raises(RuntimeError, match="Restart kernel"):
        notebook_launcher._validate_loaded_modules(runtime_root)
