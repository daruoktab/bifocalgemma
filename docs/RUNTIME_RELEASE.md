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

Publication uploads the immutable content-named ZIP while leaving `runtime/latest.zip`
and prior runtime archives unchanged.
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

## Preserve loaded host widgets, 2026-11-13

- Immutable revision: `1a2dfe7c96a1f57db193fa803b663c64d963e929`.
- Archive: `runtime/runtime-918f3dd0d5c606fe.zip`, 372,230 bytes.
- SHA-256: `918f3dd0d5c606fe2cbbbbf90235382e08f04e3d559b232ce73e8890fd0953d5`.

The latest report established a real transitive UI upgrade: active ipywidgets
8.1.8/traitlets 5.15.1 became installed 8.1.9/5.16.1. The installer now preserves
known loaded UI versions that are not explicit notebook pins using constraints
on both install calls. Constraint profiles are included in receipt identity and
validation, and a profile change cannot bypass an existing same-kernel restart
barrier. Actual incompatible versions and explicit replacements remain protected.

The focused suite passed 102 tests, including the exact reported host-version
pair and repeated setup. Nine additional widget lifecycle tests passed using
real host-version packages in an isolated target overlay. A real Linux Python
3.13/cu132 resolution selected 100 packages while retaining 8.1.8/5.15.1. No model,
dataset, training parameter or checkpoint source changed. See the
[dashboard report](DASHBOARD_WIDGET_LIFECYCLE.md) for source references and limits.
The 43 loader/notebook tests, published generator check, anonymous empty-cache
load and all four project diagnostic tools passed for this revision.

## Versionless matrix baseline, 2026-11-16

- Repository: `daruokta/bifocalgemma-2026-11-02` (public model repository).
- Immutable revision: `8ffae5e550e290b6d60ff6f5530d340106b61d09`.
- Archive: `runtime/runtime-877c2024af2a4fb4.zip`, 342,618 bytes (`104` entries).
- SHA-256: `877c2024af2a4fb4e18c6b2223b801215da05ec2a3d9279f87b0c035f9268c7c`.

This release unifies the legacy `v2/` and `v3/` implementations into canonical
versionless modules under `src/bifocalgemma/` and updates
[compatibility metadata](../configs/training_resume_compatibility.json) to
`clean_versionless_baseline_v1`. Publication left `runtime/latest.zip` and all 26
prior `runtime/*` archives unchanged, preserving running legacy Molab notebooks.
Read-back download verification from revision `8ffae5e550e290b6d60ff6f5530d340106b61d09`,
generator `--check --require-published`, strict Marimo checks across all 14 active
notebooks, static diagnostics (`ruff`, `ty`, `pyrefly`), and offline pytest suites
all passed.

## Eval overflow & transient LFS retry release, 2026-11-26

- Repository: `daruokta/bifocalgemma-2026-11-02` (public model repository).
- Immutable revision: `27468d81fd606c0fcf24f537139597d188cfd26b`.
- Archive: `runtime/runtime-a2873bf19eb153e5.zip`.
- SHA-256: `a2873bf19eb153e55fee026e7843ea07d934c7a144cda0b0fdc827e2eced84a2`.

This release adds unnumbered-line overflow fallback in evaluation context
formatting and retry handling for transient Hugging Face Hub HTTP 400 LFS
pointer propagation errors, while preserving `baseline_training_semantics_sha256`
(`50972e2a262f4297aa98ecf8178171c3dea13b124a6e56cf4e4f5f1c7440f153`) for existing
v4 runs.

## Audit remediation release, 2026-11-27

- Repository: `daruokta/bifocalgemma-2026-11-02` (public model repository).
- Immutable revision: `110cdedab8292cb832a70f52027d683f1e705c62`.
- Archive: `runtime/runtime-c05f7ccaa583f739.zip`.
- SHA-256: `c05f7ccaa583f739bbc2272dd5eed834ee50125ead45bdd41cee1b43a0a58f47`.
- Approved `source_semantics_sha256`: `23f0daf2d3df6a4fd4824146aea8c7dede1f48d61d4951a600bd143ea2b24419`
  (retaining `baseline_training_semantics_sha256`: `50972e2a262f4297aa98ecf8178171c3dea13b124a6e56cf4e4f5f1c7440f153`).

Bundles the full external-review remediation (dual `transformers` 5.22.1/5.24.1
compatibility, v4 milestone checkpoint pruning fix, `weights_only=True` hardening,
right-padded `generate()` fix, non-mutating `save_pretrained()`, per-layer
`CachedAttention` dispatch, `from_pretrained()` validation, section-aware
overflow fallback + protocol signature tracking, `recount_functional_pass_rate`
denominator fix, and `data/patches.py` header parsing fix). Anonymous read-back
verification, `--check --require-published`, and the matrix runtime loader suite
passed.

## Prompt-source byte parity & resume hotfix, 2026-11-27

- Repository: `daruokta/bifocalgemma-2026-11-02` (public model repository).
- Immutable revision: `5ca52298e8ba18129579cc8a6b9e4c9cb6e766f1`.
- Archive: `runtime/runtime-42125aab6f57b42a.zip`.
- SHA-256: `42125aab6f57b42a451175c6e1dcfe4fcf0d67f5ac37a9ee5d17e668f6a75800`.
- Approved `source_semantics_sha256`: `47ce481e8c61c3bf2d5e75d9a1d5b7e69bed4fa9d3434ff33499effca11f66f5`
  (retaining `baseline_training_semantics_sha256`: `50972e2a262f4297aa98ecf8178171c3dea13b124a6e56cf4e4f5f1c7440f153`).

Restores exact pre-format UTF-8 bytes on `src/bifocalgemma/data/context_tasks.py`
(and excludes the three raw-text-hashed `prompt_sources` modules from `ruff format`)
so resuming `e4b-it` and `2it-2it` hits the existing packed token cache and exact
`data_fingerprint`, while also accepting token-identical `data_fingerprint` variants
when `matched_target_fingerprints`, `schedule_signature`, and all 25 other
compatibility fields match (preserving future resume for `4it-4it` started on
`c05f7ccaa583f739`).
