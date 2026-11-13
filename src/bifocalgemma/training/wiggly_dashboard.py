"""Stable live widgets; durable logs and static HTML remain independent of the UI."""

import html
import math
import time
from typing import Any

from wigglystuff import (
    FloatingPanel,
    HTMLRefreshWidget,
    NestedTable,
    ProgressBar,
    ScatterLog,
    TraceView,
)

from bifocalgemma.training.telemetry import TrainingDashboard, format_duration, format_stage_display


class LastSeenWidget(HTMLRefreshWidget):
    # This clock runs in the browser, including after the Python connection dies.
    # Its wording deliberately reports staleness, never inferred kernel liveness.
    _esm = """
    export function render({model, el}) {
      function age() {
        const stamp = el.querySelector('[data-last-seen]');
        if (!stamp) return;
        const seconds = Math.max(0, Math.floor((Date.now()-Number(stamp.dataset.lastSeen))/1000));
        stamp.textContent = seconds >= 300 ? `Tidak ada kabar selama ${seconds} detik; status kernel tidak diketahui.` : `Kabar terakhir ${seconds} detik lalu.`;
      }
      function draw() { el.innerHTML = model.get('html'); age(); }
      draw(); model.on('change:html', draw);
      const timer = setInterval(age, 1000);
      return () => { clearInterval(timer); model.off('change:html', draw); };
    }
    """


def phase_category(stage):
    if stage.startswith("benchmark_upload") or stage == "remote_snapshot_verified":
        return "HF sync"
    if stage.startswith("benchmark") or stage.startswith("generation"):
        return "Benchmark"
    if stage.startswith("validation"):
        return "Validation"
    if stage.startswith("checkpoint"):
        return "Checkpoint"
    if "preflight" in stage or stage == "backend_candidate":
        return "GPU preflight"
    if "download" in stage or stage == "tokenization":
        return "Prepare data/model"
    return "Training"


class TrainingLossLog(ScatterLog):
    def _build_spec(self):
        spec = super()._build_spec()
        spec["params"] = [{"name": "view", "select": "interval", "bind": "scales"}]
        spec["encoding"]["tooltip"] = [
            {"field": "x", "title": "Optimizer step", "type": "quantitative"},
            {"field": "y", "title": "CE", "type": "quantitative", "format": ".6f"},
            {"field": "color", "title": "Split", "type": "nominal"},
        ]
        return spec


class WigglyDashboard(TrainingDashboard):
    def __init__(self, mo, *, experiment="", device=None, refresh_seconds=1.0):
        super().__init__(mo, experiment=experiment, device=device, refresh_seconds=refresh_seconds)
        self.spans: list[dict] = []
        self.plotted: set[tuple] = set()
        self.breakdown_signature = None
        self._layout_content = None
        self._widget_views: dict[str, Any] = {}
        self._floating_panel = None

    def _create_widgets(self):
        # Construct comm-backed models in the cell that actually displays them,
        # after marimo has installed its widget transport for the current session.
        self.progress_bar = ProgressBar(max_value=1, height=18)
        self.phase_bar = ProgressBar(max_value=1, height=12, color="#f59e0b")
        self.cards = HTMLRefreshWidget(html="Menyiapkan training…")
        self.activity = HTMLRefreshWidget(html="Belum ada aktivitas.")
        self.pin = LastSeenWidget(html="Menunggu kabar kernel…")
        self.loss = TrainingLossLog(
            x_label="Optimizer step",
            y_label="Cross entropy",
            color_label="Split",
            max_points=1000,
            height=280,
        )
        self.table = NestedTable(show_percent=False, initial_expand_depth=1)
        self.tasks = NestedTable(show_percent=False, initial_expand_depth=1)
        self.optimizer = HTMLRefreshWidget(html="Diagnostik optimizer muncul setiap 10 update.")
        self.trace = TraceView(title="Tahap sesi ini", height=220)

    def layout(self):
        if self._layout_content is not None:
            return self._layout_content
        mo = self.mo
        wrap = mo.ui.anywidget
        self._create_widgets()
        self._widget_views = {
            name: wrap(getattr(self, name))
            for name in (
                "progress_bar",
                "phase_bar",
                "cards",
                "loss",
                "table",
                "tasks",
                "optimizer",
                "trace",
                "activity",
                "pin",
            )
        }
        views = self._widget_views
        pieces = [
            mo.md("### Monitor training · otomatis"),
            views["progress_bar"],
            views["phase_bar"],
            views["cards"],
            views["loss"],
            mo.accordion(
                {
                    "Validasi per sumber": views["table"],
                    "Validasi per objective": views["tasks"],
                    "Optimizer": views["optimizer"],
                    "Timeline sesi": views["trace"],
                    "Log terbaru": views["activity"],
                }
            ),
        ]
        pin = views["pin"]
        panel_title = f"{self.state.get('experiment') or 'Monitor'} · status terakhir"
        if mo.running_in_notebook():
            self._floating_panel = FloatingPanel(pin, title=panel_title, width=340)
            pieces.append(self._floating_panel)
        else:
            pieces.append(pin)
        self._layout_content = mo.vstack(pieces)
        if self.recent:
            self.update_widgets()
        return self._layout_content

    def publish(self, content):
        if self._layout_content is None or self.ui_error is not None:
            return
        try:
            self.update_widgets()
        except Exception as error:
            self.ui_error = f"{type(error).__name__}: {error}"
            print(
                "Live widgets unavailable; events.jsonl and monitor.html remain available.",
                flush=True,
            )

    def update_widgets(self):
        if self._layout_content is None:
            return
        state = self.state

        def esc(value):
            return html.escape(str(value), quote=True)

        step, maximum = int(state.get("step", 0)), max(1, int(state.get("max_steps", 1)))
        self.progress_bar.max_value = maximum
        self.progress_bar.value = min(step, maximum)
        stage = state.get("stage", "preparing")
        stage_display = format_stage_display(stage, state)
        done, total = state.get("items_done", 0), state.get("items_total", 1)
        if stage.startswith("validation"):
            done, total = state.get("eval_examples_done", 0), state.get("eval_examples_total", 1)
        self.phase_bar.max_value = max(1, int(total))
        self.phase_bar.value = min(int(done), self.phase_bar.max_value)

        def number(key):
            value = state.get(key)
            return "—" if value is None else f"{value:,.4g}"

        cards = {
            "Tahap": stage_display,
            "Train CE": number("train_loss"),
            "Dev CE": number("dev_loss"),
            "Aktif teramati training": format_duration(state.get("elapsed_seconds")),
            "Sesi training ini": format_duration(state.get("segment_seconds")),
            "Aktif benchmark ini (lintas sesi)": format_duration(
                state.get("benchmark_elapsed_seconds")
            ),
            "Jeda tidak teramati": format_duration(state.get("unobserved_gap_seconds")),
            "ETA optimizer saja": format_duration(state.get("eta_train_seconds")),
            "Target token/detik": number("target_tokens_per_second"),
            "Gradient norm": number("gradient_norm"),
            "Learning rates": state.get("learning_rates", "—"),
            "GPU allocated / free (GiB)": number("gpu_allocated_gib")
            + " / "
            + number("gpu_free_gib"),
            "GPU reserved / peak allocated (GiB)": number("gpu_reserved_gib")
            + " / "
            + number("gpu_peak_allocated_gib"),
            "Microbatch / accumulation": str(state.get("batch_size", "—"))
            + " / "
            + str(state.get("accumulation", "—")),
            "Attention / query chunk": str(state.get("attention_backend", "—"))
            + " / "
            + str(state.get("query_chunk_size", "—")),
            "RAM proses / tersedia (GiB)": number("ram_rss_gib")
            + " / "
            + number("ram_available_gib"),
            "Checkpoint HF": state.get("last_remote_checkpoint", "Belum terverifikasi"),
            "Revision": str(state.get("remote_revision", "—"))[:12],
            "Upload tertunda": state.get("upload_pending", "—"),
        }
        self.cards.html = (
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px">'
            + "".join(
                f'<div style="border:1px solid #64748b;border-radius:8px;padding:10px"><small>{esc(k)}</small><br><strong>{esc(v)}</strong></div>'
                for k, v in cards.items()
            )
            + "</div>"
        )
        for row in self.history[-1000:]:
            for split, value in (
                ("train", row.get("train_loss")),
                ("dev", row.get("eval", {}).get("loss")),
            ):
                identity = (row.get("step"), split)
                if (
                    identity not in self.plotted
                    and value is not None
                    and math.isfinite(float(value))
                ):
                    self.loss.append(x=int(row["step"]), y=float(value), color=split)
                    self.plotted.add(identity)
        self.pin.html = (
            f"<strong>{esc(state.get('experiment', ''))}</strong><br>{esc(stage_display)} · {step:,}/{maximum:,}<br>"
            f"HF pending: {esc(state.get('upload_pending', '—'))}<br>"
            f'<small data-last-seen="{time.time() * 1000:.0f}"></small>'
        )
        self.activity.html = (
            "<pre style='white-space:pre-wrap'>"
            + "\n".join(
                esc(f"{r.get('timestamp_utc', '')} {r.get('stage', '')} {r.get('message', '')}")
                for r in self.recent
            )
            + "</pre>"
        )
        values = state.get("eval", {})
        signature = repr(values)
        if signature != self.breakdown_signature and values:
            for group, widget in (("by_source", self.table), ("by_task_type", self.tasks)):
                paths = {
                    f"{name} (CE {row['loss']:.5f})": {"tokens": row["supervised_tokens"]}
                    for name, row in values.get(group, {}).items()
                }
                if paths:
                    prepared = NestedTable.from_paths(
                        paths, root_name="Token validasi", show_percent=["tokens"]
                    )
                    widget.data, widget.columns = prepared.data, prepared.columns
                    widget.show_percent = ["tokens"]
            self.breakdown_signature = signature
        diagnostics = state.get("optimizer_diagnostics")
        if diagnostics:
            self.optimizer.html = "<pre>" + esc(str(diagnostics)) + "</pre>"
        category, now = phase_category(stage), time.time() * 1000
        if (
            not self.spans
            or self.spans[-1]["name"] != category
            or self.spans[-1]["end_ms"] is not None
        ):
            if self.spans:
                self.spans[-1].update(end_ms=now, status="ok")
            self.spans.append(
                {
                    "span_id": str(time.time_ns()),
                    "trace_id": "session",
                    "parent_id": None,
                    "name": category,
                    "service": None,
                    "kind": "internal",
                    "start_ms": now,
                    "end_ms": None,
                    "status": "unset",
                    "status_message": "",
                    "attributes": {},
                    "events": [],
                    "links": [],
                }
            )
        if stage in {
            "validation_complete",
            "checkpoint_complete",
            "remote_snapshot_verified",
            "pilot_completed",
            "training_completed",
            "benchmark_completed",
            "completed",
            "failed",
            "interrupted",
        }:
            self.spans[-1].update(
                end_ms=now, status="error" if stage in {"failed", "interrupted"} else "ok"
            )
        self.spans = self.spans[-200:]
        self.trace.spans, self.trace.now_ms = [dict(s) for s in self.spans], now
