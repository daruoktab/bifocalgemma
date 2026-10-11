# Dua Notebook SFT dengan Cell yang Sama

File untuk diunggah satu per satu ke Molab:

- [bifocalgemma_train_symmetric.py](../notebooks/bifocalgemma_train_symmetric.py)
- [bifocalgemma_train_asymmetric.py](../notebooks/bifocalgemma_train_asymmetric.py)

Keduanya **20 cell**, dalam urutan yang sama. **19 isi cell identik persis**.
Hanya tiga baris pada control center berbeda:

| Setting | Simetris | Asimetris |
| --- | --- | --- |
| FAMILY | symmetric | asymmetric |
| Default EXPERIMENT | BifocalGemma-E2B-E2B | BifocalGemma-E2B-E4B |
| OUTPUT_ROOT | outputs/training_symmetric | outputs/training_asymmetric |

Pilihan keluarga simetris: E2B–E2B atau E4B–E4B. Asimetris: E2B–E4B atau E4B–E2B.
Semua model memakai encoder base / decoder IT asli. Word tying dan H projection
mengikuti spec runtime dari kode model yang **sama**, sehingga tidak memerlukan
cell arsitektur berbeda. Simetris: tied ordinary words, Identity projection.
Asimetris: untied encoder words, learned H projection. PLE tetap separate.

## Optimasi memori dan monitor — 11 Oktober 2026

Default sekarang memakai layer checkpointing, tiled attention FP32 (termasuk head512),
chunked loss train/dev, dan preflight panjang maksimum tanpa optimizer update.
RUN_NAME baru: enc8k-dec2k-v2-mem1. GPU/RAM, tahap, loss chart, validasi,
checkpoint dan failure diagnostics tampil pada monitor; events/history/monitor.html
disimpan untuk inspeksi. Tidak ada quantization, pengurangan scope atau context.
Lihat [recipe dan monitoring](MEMORY_EFFICIENT_TRAINING.md).

## Cara memulai

1. Unggah salah satu file .py ke Molab; file standalone tidak perlu checkout repo.
2. Apply HF token write pada cell input/apply yang terpisah. Dataset private dan
   checkpoint gated membutuhkan akses unduh; upload aktif secara default membutuhkan
   akses write ke repo model tujuan.
3. Pilih EXPERIMENT dari keluarga notebook; ubah RUN_NAME untuk eksperimen baru.
4. Pilih DATASET_REPO_ID, DATASET_REVISION (full commit SHA), dan DATASET_COLLECTION yang sudah dipublikasikan; sesuaikan CORPUS_KIND serta LOCAL_DATA_FILE. Kurasi dijalankan terpisah di CPU, bukan otomatis saat training. Lihat [dataset releases](DATASET_RELEASES.md).
5. Default **RUN_TRAINING=True**, **HF_UPLOAD_ENABLED=True**, **MAX_STEPS=0**,
   **EPOCHS=1**: seluruh split dilatih, dev loss dihitung, kemudian artefak run selesai
   diupload. Ubah kedua flag menjadi False bila ingin menonaktifkan aksinya. Periksa
   tabel dataset READY: 40.706 train /4.524 dev sebelum training.
6. Amati preprocessing count, training logs/dev CE, lalu lokasi checkpoint/report.

Satu notebook melatih **satu pasangan pada satu run**. Untuk pasangan kedua,
ubah EXPERIMENT dan jalankan run baru. Menjalankan dua notebook di GPU yang sama
memakai resource bersama; preflight membaca free VRAM aktual, bukan mengasumsikan
kapasitas 96 GB selalu kosong. Setting training default berbeda dari micro probe
512-token terdahulu; peak di report run baru menjadi bukti yang berlaku.

## Peta cell identik

| Posisi | Peran |
| --- | --- |
| 0 | Intro |
| 1 | Control center — satu-satunya isi cell yang berbeda |
| 2 | Environment, versi, GPU dan precision |
| 3–4 | Input HF password / Apply-whoami |
| 5 | Audited config dan experiment registry |
| 6 | Batch/collator/patch CE |
| 7 | Native merged attention |
| 8 | Native model builder |
| 9 | Pretrained transfer dan attention LoRA |
| 10 | Training policy: dua profil dan meta group table |
| 11 | Frozen-pool loading, manifest dan tokenization helpers |
| 12 | Download/verifikasi koleksi HF beku dan cache tokenizer |
| 13 | Training/checkpoint/resume helpers |
| 14 | Preflight metadata/VRAM guard |
| 15 | Frozen data tokenization |
| 16 | Model load + SFT dalam satu rerun boundary |
| 17 | Cheap result summary |
| 18 | Upload artifacts ke HF Hub ({FAMILY}/{EXPERIMENT}/{RUN_NAME}) |
| 19 | Pytest-only cell |

Perubahan UI summary tidak memerlukan rerun model/training. Model dimiliki oleh
training cell dan dilepas setelah selesai. Jangan mengubah hyperparameters saat
run aktif kecuali memang hendak memulai ulang rerun boundary.

## Data dan objective

Notebook hanya menerima koleksi dataset HF beku pada commit SHA yang terkunci.
Default instruction_enc8k_dec2k_v2 mempunyai 40.706 train /4.524 dev. Tidak ada
pilihan raw-source, local_jsonl, sampling pilot atau kurasi ulang dalam notebook.
Semua IDs dan split mengikuti manifest; input before/context saja, target unified diff.
Kurasi dan pembuatan dataset dilakukan terpisah melalui CLI lokal yang dijelaskan
pada [panduan rilis](DATASET_RELEASES.md).

Split v2 menghubungkan repository, commit/seed, alias sumber, serta struktur kode.
Tidak ada re-splitting saat run. Batas encoder8.192 /paired decoder2.048 tetap;
overflow dataset beku adalah error. Dependency helper mempertahankan hash/manifest
run yang sama untuk cohort v2, sehingga perubahan default UI tidak mengganti data.


Token-weighted patch CE hanya pada target/closing turn, shift sekali, prefix/pad -100.
Gradient accumulation memakai proporsi supervised tokens per window supaya target
pendek tidak mendapat bobot berlebihan. Partial epoch window memakai denominator
aktual. Tidak ada loss tambahan atau knowledge distillation.

## Cell Training policy (setelah loader, sebelum data)

**POLICY_PROFILES** adalah sumber setting keluarga. Cell yang sama memuat dua
profil, lalu FAMILY memilih salah satunya. Edit rank/alpha dan LR/warmup di sini;
control center tidak lagi menduplikasi knob LR/rank/warmup.

| Group | Simetris | Asimetris | Default LR / warmup |
| --- | --- | --- | --- |
| encoder | Encoder attention LoRA | Encoder attention LoRA | 2e-5 / 5 steps |
| decoder | Decoder attention LoRA | Decoder attention LoRA | 2e-5 / 5 steps |
| alignment | Tidak ada, Identity | Learned H projection | 2e-5 / 5 steps |

Word/PLE tetap frozen, phase joint, rank 16 / alpha32. Profile memilih train_alignment
hanya pada asimetris. Semua trainable parameters masuk tepat satu group; frozen
parameters/aliases tidak diduplikasi. Group order encoder→decoder→alignment dan
parameter names di-sort deterministik. Tabel meta sebelum training menampilkan
parameter counts, dtype, LR, warmup, freeze dan tying policy tanpa GPU allocation.

LR per group diterapkan sebagai base_lr * min(1,(step+1)/warmup_steps); warmup0
memakai full LR langsung. Contoh eksperimen alignment LR lebih tinggi ditulis pada
POLICY_PROFILES["asymmetric"]["groups"]["alignment"]; encoder/decoder LR tidak
berubah. Semua log step menyimpan learning_rates per group. Alpha dipakai sebagai
alpha/rank scaling adapter; word training atau staged projection-only training
belum didukung oleh profile versi ini.

Baseline kedua profil sengaja memakai LR/warmup yang sama. Perbedaan LR, projection
warmup terpisah dan staged unfreezing perlu eksperimen, bukan otomatis requirement
arsitektur. Source policy dan optimizer-group signature disimpan dalam run config
serta checkpoint compatibility.

## Profil dataset dan batas token berikutnya

[Analisis Molab](MOLAB_DATA_EVALUATION_PLAN.md) memprofilkan seluruh EditPackFT: encoder
maksimum 749, decoder total maksimum 1.113 setelah contract filtering. Setelah stress
probe GPU aktual, rekomendasi S1 ialah ENCODER_LIMIT=1024 dan DECODER_LIMIT=1536;
baseline decoder-only total 2048 menampung context+target yang sama. Default notebook
lokal yang sekarang 8192/2048; baseline diberi budget 10752 untuk context dan target yang sama; final budget mengikuti distribusi token corpus penuh dan stress GPU. Lihat [full corpus](FULL_CORPUS_TRAINING.md).
NextCoder memerlukan profil terpisah: pada sampel 100 Python rows, decoder median 927,
bukan target kecil yang otomatis muat 512. Jangan mencampurnya lalu diam-diam
memotong target atau menyimpulkan 97% overflow sampel sebagai statistik populasi.

## Setting full-pool dan dependency

Attention LoRA rank 16 / alpha32, frozen BF16 base words/PLE, FP32 A/B, trainable
asymmetric H projection. AdamW dengan named parameter groups, foreach=False, weight_decay0, lr2e-5 per group, warmup 5 steps
lalu constant LR, clipping 1.0, microbatch 1, accumulation 4, satu epoch penuh. MAX_STEPS=0 menghitung budget dari panjang train split; nilai positif menjadi batas eksplisit. Dev CE dan checkpoint setiap 500 steps plus final checkpoint. Recipe ini belum merupakan optimum empiris. Pilot lama 100 steps tetap tersimpan sebagai baseline mekanik, bukan full-corpus result.

PEP723 memakai runtime Molab yang berhasil: Torch2.14.1 + CUDA13.0, Transformers5.19.0,
marimo0.25.1, safetensors0.8.0, pyarrow25.0.1 dan pytest8.4.2, Python3.13.
Actual runtime versions masuk report/checkpoint compatibility. Tidak ada pip/reinstall
Torch dalam cell. Model source revisions dan text mapping sama dengan
[pretrained probe](PRETRAINED_GPU_PROBE.md).

## Artifacts dan resume

Directory: `OUTPUT_ROOT/EXPERIMENT/RUN_NAME/`.

- data_manifest.json / run_config.json ditulis sebelum optimization.
- history.jsonl mencatat train/dev loss, LR, grad norm, token count dan CUDA peak.
- training_report.json menyimpan status/checkpoints/source/runtime.
- checkpoint-NNNNNN berisi adapters.safetensors, trainer_state.pt, manifest.json,
  dan READY.json dengan checksums. Adapter-only save mencakup seluruh trainable
  LoRA parameters serta learned asymmetric projection.

Checkpoint dibuat ke staging directory lalu rename setelah lengkap. Existing
checkpoint/run artifacts tidak ditimpa. Resume: isi **RESUME_FROM** dengan directory
checkpoint lengkap dan gunakan sumber/model/data/config yang sama. Pretrained base
weights direkonstruksi, ties/LoRA dipasang, lalu trainable weights + optimizer +
CPU/CUDA/Python RNG + sampler order/cursor dipulihkan. Checkpoints hanya pada
optimizer boundary; tidak ada pending gradient accumulation yang hilang.

Checkpoint notebook baru memakai **v2** dengan policy lengkap serta optimizer group
order/parameter names/base LR/warmup. **v1 ditolak sebelum adapter/optimizer atau run
metadata diubah**; gunakan notebook commit sebelumnya untuk melanjutkan v1. Tidak
ada silent migration optimizer state satu-group menjadi multi-group. Helper publik
fit_offline_sft(..., training_policy=None) tetap menggunakan behavior/checkpoint v1
untuk pemanggil lama.

Compatibility mencakup experiment/family/source revisions, data/tokenizer fingerprint,
policy/rank/alpha, group signature, batch/accumulation, LR/warmup/clipping, seed dan runtime version. MAX_STEPS/EPOCHS
boleh diperpanjang. Wrong compatibility, incomplete/checksum-bad checkpoint, key/shape
mismatch atau sampler invalid ditolak. Keberhasilan resume telah diuji pada native
**tiny** simetris/asimetris terhadap run uninterrupted; big-model SFT kedua notebook
belum dijalankan dari sesi lokal. Dev CE bukan functional repair benchmark.

## Upload ke Hugging Face Hub

- Target repo: `HF_REPO_ID = "daruokta/bifocalgemma-sft"`.
- Kontrol: `HF_UPLOAD_ENABLED` (default `True`).
- Struktur folder di dalam repo: dua folder tingkat atas `symmetric` dan `asymmetric`
  (mengikuti parameter `FAMILY`), di dalamnya terdapat folder experiment dan run:
  `path_in_repo = f"{FAMILY}/{EXPERIMENT}/{RUN_NAME}"`.
- Cell upload berjalan setelah training selesai (`status == "COMPLETED"`).
  Seluruh artefak run (`adapters.safetensors`, `data_manifest.json`, `run_config.json`,
  `history.jsonl`, `training_report.json`, checkpoints) diunggah menggunakan `upload_folder`.
- Token dengan izin write divalidasi sebelum proses upload dimulai; token dengan akses
  read saja akan menampilkan pesan error yang jelas dan membatalkan proses upload.

## Maintenance: sumber bersama, bukan edit dua file manual

[Generator](../tools/generate_training_notebooks.py) membaca
[template cell](../tools/templates/training_notebook.py.txt) serta shared helpers:
[data](../src/bifocalgemma/data/offline_sft.py) dan
[training/checkpoint](../src/bifocalgemma/training/offline_sft.py) dan
[policy](../src/bifocalgemma/training/policy.py).

```powershell
python tools/generate_training_notebooks.py
python tools/generate_training_notebooks.py --check
python -m marimo check --strict notebooks/bifocalgemma_train_symmetric.py notebooks/bifocalgemma_train_asymmetric.py
python -m pytest notebooks/bifocalgemma_train_symmetric.py notebooks/bifocalgemma_train_asymmetric.py -q
```

Ubah cell umum di template atau helper, lalu regenerate keduanya. Untuk perbedaan
default keluarga, edit mapping VARIANTS pada generator/control substitution. Hasil
formatter deterministik dan --check gagal bila file generated drift. File upload
standalone tetap bisa diedit di Molab; saat pairing live, gunakan runtime cm API,
kemudian sinkronkan perubahan umum ke template sebelum regeneration lokal.

Validasi lokal memakai import/graph checks tanpa mengeksekusi training, sepuluh notebook pytest tests (lima per notebook), tests diff/split,
token-weighted accumulation, checkpoint checksums/compatibility/run overwrite, serta
native tiny LoRA resume pada kedua keluarga. Model/data asli belum dilatih oleh
notebook baru ini; jalankan pilot di Molab dan catat hasilnya terpisah.

## Repository exports

Helper mendukung repository_patch_v1 di dalam koleksi HF beku saat corpus tersebut
telah dipublikasikan. Eval records dan gold-context flags ditolak. Native snapshot
identity serta pair/instruction hashes dipertahankan. Pilihan upload JSONL mentah
tidak tersedia di notebook; CLI preparer tetap tersedia untuk membangun dataset
terpisah. Kedua notebook tetap20 cell dengan19 identik.


Repository-mode corpus READY now requires v3 visible-context license evidence, in addition to licenses on patched files. Old patch-only prototype pools are rejected; rebuild under the new policy in a new output directory if an old completed pool already exists. Cached source archives are reused; no model download is required for this validation.


User-selected encoder budget (10 October 2026): ENCODER_LIMIT=8192 in both family
notebooks and CONTEXT_LIMIT=8192 in the universal notebook. Paired decoder stays
2048; baseline total is 10752 (=8192+2048+512). Named pools and run directories
now carry enc8k_dec2k, so a frozen 2k pool/checkpoint is not silently reused.
The running Molab notebook remains unchanged; upload these local notebooks for
a later run. Original-model forward/backward VRAM at this budget is unmeasured.
