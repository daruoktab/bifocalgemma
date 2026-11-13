"""Real marimo model registration must precede the initial dashboard output."""

from __future__ import annotations

import gc
import json
from collections.abc import Generator
from contextlib import contextmanager

import marimo as mo
import pytest
from marimo._ast.app import InternalApp
from marimo._messaging.types import Stream
from marimo._output.formatters.formatters import register_formatters
from marimo._runtime.context import get_context, teardown_context
from marimo._runtime.context.script_context import initialize_script_context
from marimo._types.ids import CellId_t

from bifocalgemma.training.wiggly_dashboard import WigglyDashboard


class RecordingStream(Stream):
    def __init__(self):
        self.messages = []

    def write(self, data):
        self.messages.append(json.loads(data))


@contextmanager
def widget_runtime() -> Generator[tuple, None, None]:
    register_formatters()
    stream = RecordingStream()
    initialize_script_context(InternalApp(mo.App()), stream, __file__)
    context = get_context()
    try:
        with context.with_cell_id(CellId_t("view")):
            yield context, stream
    finally:
        context.cell_lifecycle_registry.dispose(CellId_t("view"), deletion=True)
        teardown_context()


def model_opens(stream):
    return {
        message["model_id"]
        for message in stream.messages
        if message.get("op") == "model-lifecycle" and message["message"]["method"] == "open"
    }


@pytest.mark.parametrize("key", ["e2b-it", "e4b-it", "2base-2it", "4base-2it"])
def test_dashboard_created_before_context_registers_all_displayed_models(key):
    # This used to create DummyComms or context-free comms whose initial model
    # state never reached the browser; later mo.ui.anywidget wrapping did not fix it.
    monitor = WigglyDashboard(mo, experiment=key, device="cpu", refresh_seconds=0)
    assert not monitor._widget_views
    with widget_runtime() as (context, stream):
        display = monitor.layout()
        assert "Monitor training" in display.text
        opened = model_opens(stream)
        assert len(monitor._widget_views) == 10
        for view in monitor._widget_views.values():
            assert view.widget.model_id in opened
            assert context.ui_element_registry.get_object(view._id) is view


@pytest.mark.parametrize("key", ["e2b-it", "e4b-it", "2it-2it", "2base-4it"])
def test_layout_and_wrappers_remain_stable_after_gc_and_updates(key):
    with widget_runtime() as (context, stream):
        monitor = WigglyDashboard(mo, experiment=key, device="cpu", refresh_seconds=0)
        first = monitor.layout()
        ids = {name: view.widget.model_id for name, view in monitor._widget_views.items()}
        opened = model_opens(stream)
        assert monitor.layout() is first
        del first
        gc.collect()
        monitor(
            {
                "stage": "full_decoder_sft" if key.startswith("e") else "frozen_decoder",
                "step": 1,
                "max_steps": 4,
                "train_loss": 0.8,
            }
        )
        assert monitor.ui_error is None
        assert monitor.progress_bar.value == 1
        assert "0.8" in monitor.cards.html
        assert model_opens(stream) == opened
        assert {name: view.widget.model_id for name, view in monitor._widget_views.items()} == ids
        for view in monitor._widget_views.values():
            assert context.ui_element_registry.get_object(view._id) is view


def test_pre_layout_events_are_preserved_in_durable_monitor_and_first_display(tmp_path):
    monitor = WigglyDashboard(mo, experiment="e2b-it", device="cpu", refresh_seconds=0)
    monitor(
        {
            "stage": "full_decoder_sft",
            "step": 2,
            "max_steps": 4,
            "train_loss": 0.6,
            "run_dir": str(tmp_path),
        }
    )
    assert not monitor._widget_views and monitor.ui_error is None
    assert "0.6" in (tmp_path / "monitor.html").read_text(encoding="utf-8")
    with widget_runtime() as (_, stream):
        monitor.layout()
        assert monitor.progress_bar.value == 2
        assert monitor.progress_bar.max_value == 4
        assert "0.6" in monitor.cards.html
        assert len(monitor.history) == 1
        assert all(
            view.widget.model_id in model_opens(stream) for view in monitor._widget_views.values()
        )
