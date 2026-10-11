"""Read-only resource samples, durable events, and a dependency-light marimo monitor."""

import html
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def resource_snapshot(device=None):
    sample = {}
    try:
        import psutil

        sample["ram_rss_gib"] = psutil.Process().memory_info().rss / 2**30
        sample["ram_available_gib"] = psutil.virtual_memory().available / 2**30
        sample["ram_total_gib"] = psutil.virtual_memory().total / 2**30
    except ImportError:
        if Path("/proc/self/status").exists():
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    sample["ram_rss_gib"] = int(line.split()[1]) / 2**20
            for line in Path("/proc/meminfo").read_text().splitlines():
                name, value, *_ = line.split()
                if name in ("MemAvailable:", "MemTotal:"):
                    sample["ram_available_gib" if name == "MemAvailable:" else "ram_total_gib"] = (
                        int(value) / 2**20
                    )
    if device is not None and torch.device(device).type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        sample.update(
            gpu_allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
            gpu_reserved_gib=torch.cuda.memory_reserved(device) / 2**30,
            gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
            gpu_peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
            gpu_free_gib=free / 2**30,
            gpu_total_gib=total / 2**30,
        )
    return sample


def append_event(path, row):
    payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **row}
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    return payload


def format_duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "—"
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def loss_chart(history):
    rows = [r for r in history if "train_loss" in r]
    if not rows:
        return '<p class="muted">Kurva muncul setelah optimizer step pertama.</p>'
    stride = max(1, math.ceil(len(rows) / 240))
    train = [(r["step"], r["train_loss"]) for r in rows[::stride]]
    if train[-1][0] != rows[-1]["step"]:
        train.append((rows[-1]["step"], rows[-1]["train_loss"]))
    dev = [(r["step"], r["eval"]["loss"]) for r in rows if "eval" in r]
    values = [v for _, v in train + dev if math.isfinite(v)]
    if not values:
        return ""
    low, high = min(values), max(values)
    extent = max(high - low, 0.05)
    maximum_step = max(rows[-1]["step"], 1)

    def points(series):
        return " ".join(
            f"{45 + 710 * step / maximum_step:.2f},{165 - 135 * (value - low) / extent:.2f}"
            for step, value in series
        )

    circles = "".join(
        f'<circle cx="{45 + 710 * step / maximum_step:.2f}" cy="{165 - 135 * (value - low) / extent:.2f}" r="3" fill="#fb923c"><title>Dev step {step}: {value:.5f}</title></circle>'
        for step, value in dev
    )
    return f"""<svg viewBox="0 0 800 200" role="img" aria-label="Train and held-out CE by optimizer step" style="width:100%">
    <path d="M45 20 V165 H760" fill="none" stroke="#64748b"/>
    <text x="2" y="32" fill="#94a3b8" font-size="11">{high:.3f}</text>
    <text x="2" y="165" fill="#94a3b8" font-size="11">{low:.3f}</text>
    <text x="45" y="188" fill="#94a3b8" font-size="11">0</text>
    <text x="710" y="188" fill="#94a3b8" font-size="11">{maximum_step:,}</text>
    <polyline points="{points(train)}" fill="none" stroke="#38bdf8" stroke-width="1.8"/>
    <polyline points="{points(dev)}" fill="none" stroke="#fb923c" stroke-width="2"/>{circles}</svg>
    <p class="muted">● Biru: train CE per optimizer window (kurva diringkas, log lengkap). ● Oranye: held-out CE seluruh dev. Ini bukan skor repair.</p>"""


def render_dashboard(state, history, recent):
    def esc(value):
        return html.escape(str(value), quote=True)

    def number(value, digits=2):
        return "—" if value is None else f"{value:,.{digits}f}"

    step, maximum = int(state.get("step", 0)), max(1, int(state.get("max_steps", 1)))
    percent = min(100, 100 * step / maximum)
    cards = [
        ("Optimizer step", f"{step:,} / {maximum:,}"),
        ("Epoch", number(state.get("epoch_fraction"), 3)),
        ("Train CE", number(state.get("train_loss"), 5)),
        ("Dev CE", number(state.get("dev_loss"), 5)),
        ("Waktu run", format_duration(state.get("elapsed_seconds"))),
        ("ETA train saja", format_duration(state.get("eta_train_seconds"))),
        ("Target token / detik", number(state.get("target_tokens_per_second"), 1)),
        ("Gradient norm sebelum clip", number(state.get("gradient_norm"), 3)),
        ("GPU allocated", number(state.get("gpu_allocated_gib")) + " GiB"),
        ("GPU reserved", number(state.get("gpu_reserved_gib")) + " GiB"),
        ("GPU peak reserved", number(state.get("gpu_peak_reserved_gib")) + " GiB"),
        ("GPU peak allocated", number(state.get("gpu_peak_allocated_gib")) + " GiB"),
        ("GPU free", number(state.get("gpu_free_gib")) + " GiB"),
        ("RAM proses", number(state.get("ram_rss_gib")) + " GiB"),
        ("RAM tersedia", number(state.get("ram_available_gib")) + " GiB"),
    ]
    card_html = "".join(
        f'<div class="card"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>'
        for label, value in cards
    )
    dev_progress = ""
    if state.get("stage") == "tokenization":
        done, total = state.get("items_done", 0), max(1, state.get("items_total", 1))
        dev_progress = f'<p>Tokenisasi {esc(state.get("data_split", ""))}: {done:,} / {total:,}</p><progress max="{total}" value="{done}"></progress>'
    if state.get("stage", "").startswith("validation"):
        done, total = (
            state.get("eval_examples_done", 0),
            max(1, state.get("eval_examples_total", 1)),
        )
        dev_progress = f'<p>Validasi: {done:,} / {total:,} contoh · CE sementara {number(state.get("eval_running_loss"), 5)}</p><progress max="{total}" value="{done}"></progress>'
    lr = " · ".join(f"{esc(k)}={v:.3g}" for k, v in state.get("learning_rates", {}).items()) or "—"
    rows = "".join(
        f"<tr><td>{esc(row.get('timestamp_utc', ''))}</td><td>{esc(row.get('stage', ''))}</td><td>{esc(row.get('message', row.get('step', '')))}</td></tr>"
        for row in recent[-10:]
    )
    error = (
        f'<div class="failure">{esc(state["error_type"])}: {esc(state.get("error", ""))}</div>'
        if state.get("error_type")
        else ""
    )
    return f"""<style>
    .bgm {{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:16px;padding:22px}}
    .bgm h2 {{margin:0 0 8px;font-size:23px}} .bgm .muted {{color:#94a3b8;font-size:12px}}
    .bgm .badge {{display:inline-block;background:#1e3a8a;padding:5px 12px;border-radius:20px;font-size:13px}}
    .bgm .grid {{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:16px 0}}
    .bgm .card {{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px}}
    .bgm .card span {{display:block;font-size:11px;color:#94a3b8;margin-bottom:8px}} .bgm .card strong {{font-size:17px}}
    @media(max-width:800px) {{.bgm .grid {{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
    @media(max-width:450px) {{.bgm .grid {{grid-template-columns:1fr}}}}
    .bgm progress {{width:100%;height:14px;accent-color:#38bdf8}}
    .bgm table {{width:100%;font-size:12px;border-collapse:collapse}} .bgm td {{padding:6px;border-bottom:1px solid #334155;overflow-wrap:anywhere}}
    .bgm .failure {{background:#7f1d1d;padding:12px;border-radius:8px;margin:10px 0;overflow-wrap:anywhere}}
    </style><section class="bgm"><h2>BifocalGemma · Training monitor</h2>
    <p>{esc(state.get("experiment", ""))} <span class="badge">{esc(state.get("stage", "preparing"))}</span></p>
    <progress max="{maximum}" value="{step}"></progress><p class="muted">{percent:.2f}% optimizer budget · Dev, checkpoint, dan upload terlihat sebagai tahap terpisah.</p>
    <p>{esc(state.get("message", ""))}</p>
    <p class="muted">Train {esc(state.get("train_examples", "—"))} · Dev {esc(state.get("dev_examples", "—"))} · Query chunk {esc(state.get("query_chunk_size", "—"))} · Loss chunk {esc(state.get("loss_chunk_size", "—"))}</p>
    {error}{dev_progress}<div class="grid">{card_html}</div>
    <p class="muted">LR: {lr} · Microbatch {esc(state.get("microbatch", "—"))}/{esc(state.get("accumulation", "—"))}
    · Encoder {esc(state.get("encoder_tokens", "—"))} · Decoder {esc(state.get("decoder_tokens", "—"))} · Target {esc(state.get("batch_target_tokens", "—"))}</p>
    <p class="muted">Backend: {esc(state.get("attention_backend", "—"))} · Layer checkpointing: {esc(state.get("gradient_checkpointing", "—"))}
    · Batch IDs: {esc(", ".join(state.get("instance_ids", [])))}</p>
    {loss_chart(history)}<p class="muted">Checkpoint lengkap terakhir: {esc(state.get("last_checkpoint", "belum tersimpan"))}</p>
    <p class="muted">GPU peak kumulatif sejak awal segmen; ETA hanya waktu optimizer, belum termasuk evaluasi/save/upload. Nilai — berarti belum tersedia.</p>
    <details><summary>Aktivitas terbaru / diagnostics</summary><table>{rows}</table></details></section>"""


class TrainingDashboard:
    """Update a marimo cell in place. UI failure never changes training gradients."""

    def __init__(self, mo, *, experiment="", refresh_seconds=1.0, device=None):
        self.mo = mo
        self.device = device
        self.state = {"experiment": experiment, "step": 0}
        self.history: list[dict] = []
        self.recent: list[dict] = []
        self.last_refresh = 0.0
        self.refresh_seconds = refresh_seconds
        self.run_dir = None
        self.ui_error = None

    def __call__(self, event: dict):
        if "ram_rss_gib" not in event:
            event = {**resource_snapshot(self.device), **event}
        event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
        self.state.update({k: v for k, v in event.items() if k != "history"})
        if "history" in event:
            self.history = list(event["history"])
        if event.get("stage") == "optimizer_step_complete":
            self.history.append(dict(event))
        if event.get("stage") == "validation_complete":
            for row in reversed(self.history):
                if row.get("step") == event.get("step"):
                    row["eval"] = {"loss": event["dev_loss"]}
                    break
        if event.get("run_dir"):
            self.run_dir = Path(event["run_dir"])
        self.recent.append({k: v for k, v in event.items() if k != "history"})
        self.recent = self.recent[-10:]
        now = time.monotonic()
        force = event.get("stage") in {
            "run_started",
            "validation_started",
            "validation_complete",
            "checkpoint_started",
            "checkpoint_complete",
            "completed",
            "failed",
            "interrupted",
        }
        if now - self.last_refresh < self.refresh_seconds and not force:
            return
        self.last_refresh = now
        content = render_dashboard(self.state, self.history, self.recent)
        if self.run_dir is not None:
            (self.run_dir / "monitor.html").write_text(content, encoding="utf-8")
        if self.mo is not None and self.ui_error is None:
            try:
                self.mo.output.replace(self.mo.Html(content))
            except Exception as exc:
                self.ui_error = f"{type(exc).__name__}: {exc}"
                print(
                    "Monitor UI unavailable; durable logs and monitor.html remain available.",
                    flush=True,
                )
