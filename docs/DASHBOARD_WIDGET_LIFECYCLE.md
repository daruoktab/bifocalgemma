# Dashboard startup repair — 2026-11-13

The reported symptom was `Model not found for key` in the baseline monitor before
entering an HF token or pressing Start. These hexadecimal IDs identify frontend
widget models, not language-model checkpoints. Training had not started.

The main generated dashboard cell is shared by baseline and bifocal notebooks.
A clean local baseline kernel initially rendered successfully, so this change
does not claim that a baseline architecture branch caused the screenshot. Written
probes reproduced the missing-model condition when widgets were constructed before
the marimo transport/runtime context: no initial model-open messages were sent,
and later trait updates could not recreate the missing frontend state.

## Changes

- `WigglyDashboard` creates widgets when `layout()` runs in the display cell,
  rather than during object construction. It retains the wrappers, panel and
  rendered layout; repeated layout calls keep the same model IDs.
- Events received before layout still update history and the durable HTML monitor.
  Their state is applied when the display is first mounted.
- Dependency setup checks loaded UI-package versions before either READY shortcut.
  If a pinned UI package differs from its installed version, it requests a kernel
  restart instead of continuing with mixed widget implementations.
- Setup records the running UI modules' versions and source origins, including
  transitive widget dependencies. It requests a restart when installed versions
  differ from the running code or an explicit reinstall replaces loaded UI code. Installing
  another, previously unused dependency with unchanged UI versions can continue.
  The existing protections for loaded native modules are retained.
- Known loaded UI versions that are not explicitly pinned are supplied as exact
  resolver constraints to both installer calls. Explicit notebook pins retain
  precedence, and incompatible requirements must fail resolution.

The UI-stack checks cover marimo, wigglystuff, ipywidgets, anywidget and traitlets.
No model architecture, dataset, optimizer, training schedule or checkpoint format
changes are part of this repair.

## Verification

The corrected focused suite passed **102 tests**, including real marimo model notifications,
both baseline sizes, symmetric/asymmetric controls, retained widget registrations
after garbage collection, buffered events, and mocked dependency installations.
The tests never installed packages into the development environment.
The initial released loader and matrix-notebook suite separately passed **72 tests**.
Additional startup regressions cover installation with unchanged, loaded UI
versions, repeated Run All, the real matrix launcher, transitive UI changes,
explicit reinstalls and a changed UI becoming ready after restart.
The launcher test uses the actual imported marimo module both with and without
distribution metadata in the selected environment. Missing metadata does not
mean that marimo is absent from the running host kernel.
Nine additional widget tests passed using an isolated overlay with real
ipywidgets 8.1.8, traitlets 5.15.1, wigglystuff 0.5.36 and anywidget 0.11.0.
The conda `ai` package installation was not modified by that overlay.

A browser session rendered `e2b-it`, `e4b-it` and a bifocal control together with
an empty token form and an unpressed Start button. No missing-model errors were
observed. The precise remote Molab session from the screenshot was not instrumented.

Pyrefly (`--min-severity=warn`), ty, Ruff (`--fix`) and marimo strict checks on all
ten matrix notebooks passed in conda `ai`. The generator's
`--check --require-published` check also passed.

The updated runtime is published at immutable Hub revision
`1a2dfe7c96a1f57db193fa803b663c64d963e929`, archive
`runtime/runtime-918f3dd0d5c606fe.zip` (372,230 bytes), SHA-256
`918f3dd0d5c606fe2cbbbbf90235382e08f04e3d559b232ce73e8890fd0953d5`.
An anonymous download from an empty local archive/cache passed checksum
verification using the baseline notebook's actual loader and a temporary
marimo-style cell path. Imports resolved to the verified archive, and both
baseline dashboards registered all ten displayed widget models from that code.

The training-semantics hash before and after this UI repair is identical:
`79b2a4e2de7774c22d7bd331f3b0e549c2bb24c7050b26789a3604fa770da8ff`.
Local receipts and written probes are in `outputs/dashboard_validation`.

## Correction of the initial restart guard

The initial `3350bd79...` runtime treated any loaded UI module as requiring a
restart after any install. Since marimo is already loaded by notebook startup,
this could also stop a newly uploaded bifocal notebook whose UI versions had
not changed. Fifteen unchanged-UI cases reproduced that regression before the
correction and passed afterward. The widget repair itself is retained.

Use the latest notebook for new sessions. Already running bifocal notebooks can
keep their pinned runtimes. A kernel already marked restart-required by the
previous release still needs its own restart: pending receipts are preserved,
because the older receipt does not record which loaded packages were replaced.
The correction does not reset other kernels or delete setup/training artifacts.

## Host UI without notebook-environment metadata

A subsequent fresh-upload report for runtime `d2010a35...` showed `uv` adding
marimo to `/tmp/uv-venv` while the notebook was already running marimo. The previous
guard compared distribution metadata before and after installation. Host-loaded
marimo with version `0.25.2` but no venv metadata was consequently interpreted as
changing from `None` to `0.25.2`, even when the running UI code did not change.
Six metadata-absent UI/launcher cases reproduced the failure before this correction.

The guard now compares the loaded module's version to the installed version.
Adding matching metadata is allowed. After installation, different or unknown UI
versions, explicit reinstalls and loaded native modules still require a restart.
Receipts and restart errors name the reasons, running versions and source origins.
Existing READY fast paths retain their prior behavior for unknown UI versions;
known stale UI versions are still rejected before either shortcut.

The user's remote log did not include loaded module versions or native-module
state. The new tests cover the reproduced false-positive mechanism; they do not
establish that every possible host-version mismatch is resolved without restart.
The live Molab session was not accessible for direct verification.

## Preserve the live host's compatible widget versions

The next report supplied the missing live-version evidence: ipywidgets 8.1.8 and
traitlets 5.15.1 were active from `/usr/local/lib/python3.13/site-packages`, while
the installer selected 8.1.9 and 5.16.1 in the notebook venv. In that case the
restart guard correctly detected a real change; the installer had unnecessarily
introduced the incompatibility.

The current installer constrains known, loaded, unpinned UI packages to their
live versions. The temporary constraint file is passed to both CUDA and PyPI
installer calls. These are additive [uv constraints](https://docs.astral.sh/uv/pip/compile/#adding-constraints),
not dependency overrides, and do not force installation of an otherwise unused
package. Constraint versions are bound into the receipt identity and checked
against installed metadata when present. A same-session pending restart from any
earlier constraint profile still blocks setup before either READY shortcut.

Two launcher tests reproduced the exact reported 8.1.8/5.15.1 upgrade before the
fix and now preserve both versions across setup and repeated Run All. A real uv
compile for all Linux Python 3.13/cu132 notebook requirements resolved 100 packages
with that exact host pair. This was a resolver check, not a model training run.
Inputs and the resolved requirements are retained under
`outputs/dashboard_validation/molab-linux*`; runtime UI-version receipts are in
`host_widget_overlay_versions.json` and `host_widget_overlay.xml` there.
