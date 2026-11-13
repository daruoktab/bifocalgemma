# Releasing standalone matrix notebooks

The ten matrix notebooks load a hash-verified source ZIP. Pushing their Python
files to GitHub does not upload that ZIP: `notebooks/runtime/*.zip` is ignored.
A `LOCAL-PREVIEW` manifest works only with a matching local ZIP or verified cache.
It cannot be distributed as a standalone Molab notebook.

## Local development

Use the project's conda `ai` environment in PowerShell:

```powershell
conda run --no-capture-output -n ai python tools/generate_full_matrix_notebooks.py
conda run --no-capture-output -n ai python tools/generate_full_matrix_notebooks.py --check
```

The loader checks both the local ZIP and its cache before trying the remote
source. An unpublished preview without either artifact raises an explicit
publication error. Downloaded bytes still must match the expected SHA-256, and
unsafe ZIP paths are rejected before imports. Workspace source paths do not
override the verified runtime.

## Publication and release verification

After authorizing publication of the source payload to the configured Hugging
Face runtime repository:

```powershell
conda run --no-capture-output -n ai python tools/generate_full_matrix_notebooks.py --publish
conda run --no-capture-output -n ai python tools/generate_full_matrix_notebooks.py --check --require-published
conda run --no-capture-output -n ai python -m pytest tests/test_matrix_runtime_loader.py -q
```

Publication uploads the immutable content-named ZIP and updates `runtime/latest.zip`.
The generator forces an anonymous read-back of the immutable ZIP, verifies its
checksum, and pins all ten notebooks to the returned commit revision. The release
check requires a full immutable revision, so a preview cannot pass it.

The regression suite executes the rendered loader with a temporary marimo-style
cell filename, local and cached archives, mocked remote transport, corrupt bytes,
and unsafe paths. It also rejects checked-in matrix notebooks that still refer to
`LOCAL-PREVIEW`. For a released notebook, verify the anonymous remote-load path as
well as local extraction. Restart the notebook kernel after loading an updated
notebook so previously imported project modules cannot remain active.

## Verified release, 2026-11-12

- Repository: `daruokta/bifocalgemma-2026-11-02` (public model repository).
- Immutable revision: `55cdfa8bfbb28ee785f37db0787aca95c89c431f`.
- Archive: `runtime/runtime-cf70346b7c90c626.zip`, 370,699 bytes.
- SHA-256: `cf70346b7c90c626c7c238fcea0b540e37b9425e9ea64179536b568948270556`.

The publication was explicitly approved by the user. Anonymous read-back and the
actual rendered loader succeeded from an empty cache with a temporary marimo cell
path. Imports resolved to the verified archive. All 62 focused bootstrap/notebook
tests passed. Local receipts are under `outputs/runtime_release_validation`.

## Dashboard update, 2026-11-13

- Immutable revision: `3350bd79ca6b44ffdac9d9835a1940d6d7d99742`.
- Archive: `runtime/runtime-a53c0fe0f2192362.zip`, 371,262 bytes.
- SHA-256: `a53c0fe0f21923622a1ab63893ae1012a775358e43e07304487eeae5b3c6f8f4`.

This release repairs widget initialization and loaded UI dependency checks;
training semantics retain the previous release's identity. All ten notebooks
are pinned to the new immutable revision. Anonymous loading from an empty cache
verified the archive, project imports and all ten widget registrations for both
baseline sizes. The focused dashboard/dependency suite passed 61 tests, and the
released loader/matrix-notebook suite passed 72 tests. The four project
diagnostic tools passed. Details and adoption instructions are in
[the dashboard repair report](DASHBOARD_WIDGET_LIFECYCLE.md); local receipts are
under `outputs/dashboard_validation`.

## Dependency guard correction, 2026-11-13

- Immutable revision: `b8f92415c71511ffaecf37b0d67c4840729678b3`.
- Archive: `runtime/runtime-d2010a35f5efb6fe.zip`, 371,463 bytes.
- SHA-256: `d2010a35f5efb6fe5b250e3999c4150027082c04a3fff889324c0c8796e18a3a`.

The preceding dashboard release's UI guard was too broad: installing an unrelated
dependency could demand a restart solely because marimo was already loaded.
The corrected guard compares loaded UI packages' installed versions before and
after installation, including transitive widget dependencies. Unchanged UI no
longer forces a restart for an ordinary install. Changed UI, explicit reinstalls
and the existing native-module protections still require one when appropriate.

Fifteen unchanged-UI regression cases failed before the fix. The corrected focused
suite passed 83 tests, including the actual launcher and repeated setup. Model
and widget source, training semantics, dataset and parameter settings are unchanged
from the preceding dashboard release. Pending restart receipts from that release
are not deleted automatically; see [the repair report](DASHBOARD_WIDGET_LIFECYCLE.md).
The current loader/notebook release checks passed 43 tests; anonymous loading and
all ten widget registrations in each baseline passed from an empty cache.
Pyrefly, ty, Ruff and marimo strict checks passed, as did the generator's
`--check --require-published` check.

## Inherited host UI correction, 2026-11-13

- Immutable revision: `f6f208e4ca09ce42cc4dce11a293e8a3460fb5d6`.
- Archive: `runtime/runtime-e89c07583861cfd8.zip`, 371,742 bytes.
- SHA-256: `e89c07583861cfd8a0564ecca7f2bb80b6e8be1bf36a7ac3cc826480cb4cc626`.

The preceding correction still compared distribution metadata instead of the
running UI version. On a host-provided marimo kernel, adding matching-version
metadata to `/tmp/uv-venv` could therefore trigger a false restart. The current
guard records loaded UI versions/origins and compares those versions to the
installed distributions. Restart receipts and errors include specific reasons.
Model, widget, dataset and training parameter source is unchanged.

Six new metadata-absent regression cases failed before the fix; the focused
startup/dependency/widget suite now passes 99 tests. The launcher regression uses
the actual marimo module with absent target-environment metadata. Actual UI
changes, unknown UI versions after installation, explicit reinstalls and loaded
native modules retain their restart protection. See the [repair report](DASHBOARD_WIDGET_LIFECYCLE.md)
for validation boundaries and pending-receipt behavior.
The 43 loader/notebook release checks and all four project diagnostic tools
passed. Anonymous loading from an empty cache verified the current archive,
imports and baseline widget registrations; the generator's published check passed.
