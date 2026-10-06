# Direct SFT: Recipe Utama dan Slot Implementasi

## Recipe aktif

Bangun target encoder base + decoder IT dengan merged attention sejak initialization.
Terapkan tied ordinary word embeddings pada simetris sebelum optimizer dibuat;
asimetris untied lintas encoder/decoder, PLE tetap terpisah. Latih pada context/issue
→ patch dengan supervised cross-entropy. Teacher network, conversion checkpoint
dan distillation tidak diperlukan pada jalur utama.

Target tuning adalah offline instruction-to-patch SFT. Pilot menggunakan EditPackFT
(old_contents + instruction → diff before/after); CommitPackFT Python opsional
setelah overlap audit. Tidak ada RL, reward model, agent rollout, execution feedback,
VM atau Docker dalam data preparation/training. Function tests dilakukan terpisah
pada tahap evaluasi ketika runner tersedia.

Titik awal decoder IT membantu menyediakan kemampuan mengikuti instruksi, tetapi
retention sesudah perubahan attention/shared encoder words perlu diukur.
Fine-tuning tidak dianggap berhasil hanya karena forward/backward tidak error.

## Urutan blok

```text
C01 spec → D04 dataset/overlap → D03 tokenizer → D02 batch
       → M04 initialization/M02 ties → M05 native model
       → T03 profiling → T02 PEFT/optimizer → T01 CE → S01 checkpoint
```

[train_direct_sft](../src/bifocalgemma/training/trainer.py) adalah placeholder
yang fail pada T02. Ia belum membangun optimizer, memuat dataset atau model.
[Config contoh](../configs/train.example.json) sengaja memberi null pada
hyperparameters yang belum dipilih dari profiling; teacher_required=false.
Policy config mencatat reinforcement_learning=false, execution_feedback=false,
requires_vm=false, requires_docker=false dan primary_dataset=nuprl/EditPackFT.
Trainer tetap placeholder; metadata policy bukan implementasi training loop.

## Loss yang tersedia

[patch_cross_entropy](../src/bifocalgemma/training/objectives.py) menerima logits
[B,L,V] dan labels [B,L] dari collator. Loss hanya pada patch/closing target.
Prefix dan pad berlabel -100. Jika semuanya ignored, hasilnya zero differentiable,
bukan NaN. Cross entropy dihitung pada float32 logits untuk reference.

Logits/labels telah aligned. Native model wrapper M05 harus mencegah double shift
dari internal Hugging Face causal-LM loss. Response/trajectory losses, localization
loss dan distillation bukan bagian objective default sekarang.

## Profiling dan trainable scope

T03 harus mengukur hardware, precision, micro batch, encoder length, decoder
prefix/target length, weights resident/offload, activations, optimizer states,
throughput serta peak allocated/reserved VRAM. Profiling native baru dapat dilakukan
setelah M04/M05; angka loss smoke bukan profile model besar.

Setelah profiling, tetapkan PEFT/full/frozen policy, learning rate, batch,
gradient accumulation dan max_steps dalam run manifest. Untuk pilot hemat,
pertimbangkan adapters pada merged q/k/v/o dan encoder, serta asymmetric alignment
projection. Semua trainable parameters harus dicatat; jangan secara tidak sengaja
membekukan seluruh parameter pada simetris yang adapter H-nya Identity.

Word sharing membuat gradient encoder/decoder bertemu pada Parameter yang sama.
Optimizer memakai parameter unik. PLE memiliki policy freeze/offload tersendiri;
tying ordinary words tidak menghilangkan memory PLE. Precision/quantization,
data/token budget dan trainable scope harus sebanding atau dilaporkan sebagai
trade-off pada kedua baseline dan keempat pasangan.

## Checkpoint dan reproducibility

S01 menyimpan model spec, architecture version, source checkpoint revisions,
tokenizer/Jinja hashes, dataset manifests, seed, trainable parameter list,
precision/offload/PEFT config, optimizer/scheduler state dan alias metadata.
Resume membangun final architecture dan word aliases sebelum state/optimizer load.
Cross encoder/decoder tying tidak selalu dipulihkan oleh native decoder flag saja.

M04 harus mencatat encoder body/PLE base, decoder body/PLE IT, symmetric shared
words IT dan newly initialized asymmetric H projection. Tidak ada post-hoc merge
dari dua softmax atau perataan bobot yang diperlakukan sebagai transfer exact.

## Opsi yang sengaja tidak menjadi default

[T04](../src/bifocalgemma/training/distillation.py) adalah placeholder research.
Online distillation menambahkan teacher forward, memory dan loss/alignment logic;
offline targets mengalihkan sebagian biaya ke inference awal dan storage.
Aktifkan hanya setelah ada pertanyaan/hasil yang membenarkannya.

GatedCrossAttentionBridge tetap reference terpisah untuk ablation. Training
separated → conversion → recovery/distillation adalah ide yang ditunda, bukan
syarat direct SFT. [Log riset](RESEARCH_LOG.md) merekam keputusan ini.

## Yang bisa dicoba sekarang

```powershell
conda activate ai
$env:PYTHONPATH = 'src'
python -m bifocalgemma validate-config
python -m bifocalgemma smoke --experiment all --device cpu
python -m bifocalgemma train --experiment BifocalGemma-E2B-E2B
```

Dua perintah pertama berjalan offline. smoke melakukan satu forward/backward
synthetic tanpa optimizer atau data riil. train sengaja exit 2 dengan blok T02
dan tautan acceptance criteria; itu bukan training yang sukses.

## Pembaruan validasi lokal — 6 Oktober 2026

[Native local validation](NATIVE_LOCAL_VALIDATION.md) menjelaskan M07 native tiny,
D03 offline tokenizer dan D06 snapshot/patch contract. M04 pretrained transfer,
D04 pengambilan data/overlap audit, serta trainer/harness produksi belum selesai.
SWE-smith menyimpan bug-inducing patch: target repair perlu arah sebaliknya pada
snapshot buggy yang tepat. Direct SFT tetap hipotesis; T5Gemma 2 menggunakan tahap
adaptasi UL2 sebelumnya. PLE + tied words BF16 sendiri 9,5/11,75 GiB pada pasangan
simetris, sehingga linear-only QLoRA tidak memberi jaminan muat di GPU lokal 6 GiB.

## Probe asli pada Molab

[Pretrained GPU probe](PRETRAINED_GPU_PROBE.md) tersedia sebagai notebook standalone
terpisah. Ia memuat pinned base/IT text weights, melakukan attention-LoRA/AdamW
microsteps, dan membandingkan 1–4 independent resident models. Loader/optimizer
helpers diuji pada tiny fixtures; GPU96/pretrained measurements dijalankan pengguna.
Ini prototype profiling, bukan penyelesaian production trainer/cache/checkpoint.

## Standalone SFT notebooks dengan cell seragam

[Training notebooks](TRAINING_NOTEBOOKS.md) menyediakan jalur pilot paired-file
editing → pretrained attention LoRA SFT → held-out CE → adapter/optimizer/RNG/cursor
checkpoint resume. Keluarga simetris/asimetris memakai 18 cell yang sama, dengan
17 isi cell identik. Shared data/training helpers telah diuji pada tiny fixtures;
RUN_TRAINING default False. CLI production train_direct_sft tetap placeholder.
