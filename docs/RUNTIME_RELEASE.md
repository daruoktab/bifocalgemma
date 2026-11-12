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
