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
- Setup records installed versions of loaded UI modules before installation,
  including transitive widget dependencies. It requests a restart when those
  versions change or an explicit reinstall replaces loaded UI code. Installing
  another, previously unused dependency with unchanged UI versions can continue.
  The existing protections for loaded native modules are retained.

The UI-stack checks cover marimo, wigglystuff, ipywidgets, anywidget and traitlets.
No model architecture, dataset, optimizer, training schedule or checkpoint format
changes are part of this repair.

## Verification

The corrected focused suite passed **83 tests**, including real marimo model notifications,
both baseline sizes, symmetric/asymmetric controls, retained widget registrations
after garbage collection, buffered events, and mocked dependency installations.
The tests never installed packages into the development environment.
The initial released loader and matrix-notebook suite separately passed **72 tests**.
Additional startup regressions cover installation with unchanged, loaded UI
versions, repeated Run All, the real matrix launcher, transitive UI changes,
explicit reinstalls and a changed UI becoming ready after restart.

A browser session rendered `e2b-it`, `e4b-it` and a bifocal control together with
an empty token form and an unpressed Start button. No missing-model errors were
observed. The precise remote Molab session from the screenshot was not instrumented.

Pyrefly (`--min-severity=warn`), ty, Ruff (`--fix`) and marimo strict checks on all
ten matrix notebooks passed in conda `ai`. The generator's
`--check --require-published` check also passed.

The updated runtime is published at immutable Hub revision
`b8f92415c71511ffaecf37b0d67c4840729678b3`, archive
`runtime/runtime-d2010a35f5efb6fe.zip` (371,463 bytes), SHA-256
`d2010a35f5efb6fe5b250e3999c4150027082c04a3fff889324c0c8796e18a3a`.
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
