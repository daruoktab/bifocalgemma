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
- Installing dependencies while a selected UI stack is already loaded also
  requires a restart. The existing receipt prevents reinstall loops afterward.

The UI-stack checks cover marimo, wigglystuff, ipywidgets, anywidget and traitlets.
No model architecture, dataset, optimizer, training schedule or checkpoint format
changes are part of this repair.

## Verification

The focused suite passed **61 tests**, including real marimo model notifications,
both baseline sizes, symmetric/asymmetric controls, retained widget registrations
after garbage collection, buffered events, and mocked dependency installations.
The tests never installed packages into the development environment.
The released loader and matrix-notebook suite separately passed **72 tests**.

A browser session rendered `e2b-it`, `e4b-it` and a bifocal control together with
an empty token form and an unpressed Start button. No missing-model errors were
observed. The precise remote Molab session from the screenshot was not instrumented.

Pyrefly (`--min-severity=warn`), ty, Ruff (`--fix`) and marimo strict checks on all
ten matrix notebooks passed in conda `ai`. The generator's
`--check --require-published` check also passed.

The updated runtime is published at immutable Hub revision
`3350bd79ca6b44ffdac9d9835a1940d6d7d99742`, archive
`runtime/runtime-a53c0fe0f2192362.zip` (371,262 bytes), SHA-256
`a53c0fe0f21923622a1ab63893ae1012a775358e43e07304487eeae5b3c6f8f4`.
An anonymous download from an empty local archive/cache passed checksum
verification using the baseline notebook's actual loader and a temporary
marimo-style cell path. Imports resolved to the verified archive, and both
baseline dashboards registered all ten displayed widget models from that code.

The training-semantics hash before and after this UI repair is identical:
`79b2a4e2de7774c22d7bd331f3b0e549c2bb24c7050b26789a3604fa770da8ff`.
Local receipts and written probes are in `outputs/dashboard_validation`.

When adopting the updated runtime, reload the **baseline** notebook and restart
that kernel once. An already running bifocal notebook can keep its pinned runtime.
