# Desain Studi: Empat Pasangan Base/IT dan Dua Baseline

## Tujuan dan peran checkpoint

Pertanyaan utama: pada keluarga Gemma 4 Edge, apakah kapasitas encoder atau decoder
lebih membantu pemahaman codebase dan repair lintas file, pada kualitas/biaya yang
diukur? Notasi E2B–E4B IT berarti encoder base E2B dan decoder IT E4B.

| Canonical ID | Encoder | Decoder | Cross-component word tying |
| --- | --- | --- | --- |
| BifocalGemma-E2B-E2B | google/gemma-4-E2B | google/gemma-4-E2B-it | Ya, anchor IT |
| BifocalGemma-E4B-E4B | google/gemma-4-E4B | google/gemma-4-E4B-it | Ya, anchor IT |
| BifocalGemma-E2B-E4B | google/gemma-4-E2B | google/gemma-4-E4B-it | Tidak |
| BifocalGemma-E4B-E2B | google/gemma-4-E4B | google/gemma-4-E2B-it | Tidak |
| Gemma-4-E2B-IT | Decoder-only context | google/gemma-4-E2B-it | Tidak berlaku |
| Gemma-4-E4B-IT | Decoder-only context | google/gemma-4-E4B-it | Tidak berlaku |

[Registry C01](../src/bifocalgemma/config.py) dan [matrix JSON](../configs/experiments.json)
menyimpan IDs/revisions/policy. T5Gemma/T5Gemma 2 tetap related work, tanpa menjadi
kandidat tambahan benchmark utama.

## Recipe yang dipakai

Target architecture **merged dari initialization**, direct SFT issue/context → patch,
tanpa teacher wajib. Simetris men-sharing ordinary words sejak awal; PLE terpisah.
Encoder body/PLE berasal dari base, decoder body/PLE berasal dari IT. Pada simetris,
encoder ordinary words berasal dari shared IT table, sehingga provenance bukan
seluruh encoder berasal dari base. Asimetris memakai Henc → Ddec adapter, tetapi
word tables tetap terpisah.

[T5Gemma](https://arxiv.org/abs/2504.06225) menyediakan dasar adaptation/asymmetry;
[T5Gemma 2](https://arxiv.org/abs/2512.14856) menyediakan merged-attention/tied-word
inspiration. Current code memiliki reference block, bukan native model siap training.
[Arsitektur](ARCHITECTURE.md) menjelaskan target dan gap.

## Kontribusi yang ditargetkan

1. Adaptasi encoder-decoder Edge untuk codebase understanding dan repair lintas file.
   Encoder-decoder coding sendiri sudah punya prior art seperti
   [CodeT5](https://arxiv.org/abs/2109.00859); novelty harus pada metode/bukti domain.
2. Recipe mixed initialization base body encoder / IT decoder dengan explicit
   symmetric IT-word sharing, dan sumber parameter yang bisa dilacak.
   Keunggulan initialization tidak terisolasi tanpa kontrol recipe lain.
3. Studi alokasi kapasitas dalam empat pasangan, dengan kualitas, latency,
   physical/trainable parameters dan memory dilaporkan bersama.

## Pertanyaan yang dijawab

- **RQ1:** bandingkan pasangan dengan baseline yang ukuran decodernya sama.
  E2B–E2B IT versus E4B-IT adalah perbandingan khusus kualitas/biaya yang menarik.
- **RQ2:** duel E4B–E2B IT versus E2B–E4B IT; keduanya cross-component untied.
  Ini perbandingan utama alokasi kapasitas pada jumlah parameter text awal sama.
- **RQ3:** laporkan empat sel dan apakah manfaat reader/generator berubah menurut
  pasangan. E2B/E4B juga berbeda depth, width, MLP dan KV layout; hasil berlaku
  untuk pilihan varian keluarga ini, bukan hukum universal parameter count.

Jika S22/S24/S42/S44 adalah skor dengan protokol identik, ringkasan encoder
`((S42−S22)+(S44−S24))/2`, decoder `((S24−S22)+(S44−S42))/2`,
dan interaksi `S44−S42−S24+S22` dapat dilaporkan sebagai efek konfigurasi.
Pada default, tying hanya di diagonal; skor tersebut tidak mengisolasi efek
kapasitas/interaksi murni dari embedding sharing. Untuk klaim murni, perlukan
symmetric untied controls dengan architecture/training budget sama.

Controls adalah ablation di dalam keluarga yang sama, bukan keluarga baru pada
benchmark utama. Pilot boleh pada subset, tetapi empat pasangan tetap target
paper; subset dan konfigurasi yang belum dijalankan harus disebut dengan jelas.

## Kontrol dan pengukuran

- Pin data/source revisions, split/task IDs dan near-duplicate/commit overlap audit.
- Samakan context selection/retrieval, sampling, token/tool budget, tools dan harness.
- Untuk atribusi arsitektur, gunakan data/kurikulum dan token adaptation budget
  sebanding bagi baseline. Baseline IT zero-shot diberi label terpisah sebagai
  perbandingan peningkatan sistem, bukan efek arsitektur terisolasi.
- Gunakan native decoder GQA/KV-sharing secara metodologis konsisten. Adapter H
  satu per model, bukan jumlah berbeda per decoder layer yang tidak dicatat.
- Laporkan precision/quantization, physical/trainable parameters, training tokens/
  FLOPs/wall time, cold/warm latency, throughput, peak VRAM dan failures.
- Gunakan per-task paired outcomes/uncertainty, seed yang dicatat, serta protocol
  yang dibekukan sebelum final Verified evaluation.

## Parameter dan interpretasi kemenangan

Dari native counts sebelum perubahan attention, E2B–E2B sekitar 9,26B text parameters
untied atau 8,85B setelah ordinary word tying; E4B-IT tunggal sekitar 7,46B.
Kemenangan E2B–E2B IT tetap menarik, tetapi bukan iso-parameter fisik/iso-memory.
Asimetris sekitar 12,09B sebelum adapter. [Audit JSON](GEMMA4_JSON_AUDIT.md)
dan [arsitektur](ARCHITECTURE.md) memberi asumsi serta batas hitungan.

Primary validation awal adalah held-out offline editing. Functional repair
evaluation diarahkan ke Python tasks setelah scorer tersedia; SWE-bench Lite/
Verified ditunda. Klaim lintas file membutuhkan static multi-file held-out records.
CrossCodeEval tetap completion diagnostic. [Dataset](DATASETS_AND_BENCHMARKS.md) dan
[evaluasi](EVALUATION.md) menjelaskan scoring dan denominator.

## Jalur perkembangan

Urutan: reference → native tiny M07 + offline tokenizer D03 + patch contracts D06
→ pretrained M04/M05 + verified real D04 → profiling/PEFT/SFT → checkpoint
→ generation/harness → four-pair analysis.
[Build guide](BUILD_GUIDE.md) menyebut dependency dan acceptance tiap slot.

Gated separated attention, separated-to-merged conversion, teacher training dan
distillation merupakan opsi ablation/riset lanjutan. Mereka tidak menjadi default
atau syarat untuk melacak asal parameter. [Log riset](RESEARCH_LOG.md) menjaga
riwayat keputusan tanpa mencampurnya dengan recipe aktif.
