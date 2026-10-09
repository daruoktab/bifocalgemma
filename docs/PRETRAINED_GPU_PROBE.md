# Molab: Checkpoint Asli dan Probe VRAM Training Beberapa Model

Unggah [bifocalgemma_pretrained_probe.py](../notebooks/bifocalgemma_pretrained_probe.py)
ke Molab. Notebook **standalone 16 cell** memuat text weights asli dari encoder base
serta decoder IT dan mengukur micro training steps. Pengguna menjalankannya pada GPU
Blackwell 96 GB sendiri. Sesi lokal hanya memeriksa helper dan guard; belum ada hasil
VRAM checkpoint asli atau klaim jumlah model yang pasti muat.

## Setting awal

Control center sudah menetapkan:

| Setting | Default |
| --- | --- |
| RUN_PRETRAINED / RUN_CONCURRENCY | True / True; GPU/headroom guard sebelum download |
| Single pilot | BifocalGemma-E2B-E2B |
| Concurrent list | E2B–E2B, E2B–E4B, E4B–E2B, E4B–E4B |
| Checkpoint precision | BF16, tanpa quantization atau CPU offload |
| Training policy | Attention-only LoRA, rank16 / alpha32; asymmetric projection trainable |
| Optimizer / steps | torch.optim.AdamW, lr1e-4, foreach=False; 2 steps per model |
| Batch / lengths | 1, encoder512 actual tokens, decoder max128 |
| Headroom | 8 GiB tambahan di atas weights + estimated gradients/Adam states |
| Cache / reports | outputs/hf_cache / outputs/marimo_pretrained |

Data satu contoh arithmetic repair **sintetis**, ditokenisasi dengan tokenizer asli
base/IT dan Jinja IT. Source kode sebelum fix dan issue masuk context, target hanya
patch. Komentar sintetis menambah encoder context sampai panjang yang diminta lalu
cropping token dilakukan secara eksplisit untuk **memory stress**, bukan recipe
training dataset riil. Decoder overflow ditolak. Untuk mencoba baseline decoder-only
pada context512, naikkan MAX_DECODER_TOKENS (misalnya768), karena context ada di prefix.

GPU/VRAM/version dideteksi runtime. Notebook di CPU/laptop 6 GiB menampilkan
SKIPPED_HEADROOM sebelum checkpoint download. Flag True sendiri tidak membuat
notebook mengasumsikan hardware pengguna tersedia.

## Checkpoint dan transfer

Revisions dipin sesuai audit:

| Source | Revision |
| --- | --- |
| google/gemma-4-E2B | d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f |
| google/gemma-4-E4B | 411aa17b749aa952df1359d2dcea73917a544d9a |
| google/gemma-4-E2B-it | 3e22461f65e89153144f8adb70e3b8c2cc9845a7 |
| google/gemma-4-E4B-it | ee0ef6023621cff504d758262d4e04895a5af4a2 |

Repo metadata/header yang diperiksa menunjukkan satu model.safetensors,
text prefix **model.language_model.**, serta LM head tied ke words (tidak ada
lm_head tensor terpisah). Config/tokenizer/Jinja hashes dicocokkan dengan audit.
Checkpoint download memakai Hugging Face Hub dan pinned revision; HF token dimasukkan lewat password form pada cell **2a**, lalu klik **Apply HF token**.
Cell **2b** memverifikasi whoami dan menerapkan token sebagai dependency eksplisit
semua Hub/tokenizer calls. Read token cukup; tidak ada upload dalam probe.
Widget kosong memakai HF_TOKEN dari environment jika tersedia; tanpa keduanya,
loader memakai akses anonim eksplisit untuk checkpoint publik. Token invalid
menghentikan downstream loaders. Nilai token tidak ditulis ke report atau cache
login; report hanya menyimpan mode/source/username/verified.

[transfer_text_checkpoint](../src/bifocalgemma/models/checkpoint_transfer.py)
memeriksa seluruh required text keys/shapes sebelum copy. Unexpected text keys
menimbulkan error, kecuali K/V/norm pada native shared-KV layers yang memang tidak
memiliki proyeksi sendiri. Semua nontext tensors diabaikan dengan count pada report.
Copy CPU mmap → destination CUDA dilakukan langsung, tanpa second GPU copy tabel PLE.

Encoder body/PLE base; decoder body/PLE/words IT. Words encoder simetris di-skip saat
transfer karena sudah alias decoder IT. PLE tidak ditie. H projection asimetris baru
random dan trainable. Source revisions, jumlah tensor mapped, skipped words dan
ignored shared-KV tensors dicatat. Semua actual text parameters loaded; bukan random
fallback jika file/key/shape salah.

File weights upstream monolithic: vision/audio payload **tetap ikut terunduh/cache**
walaupun hanya text tensors dimuat ke GPU. Empat source checkpoints dapat membutuhkan
sekitar 50 GB disk cache; pemakaian GPU text-only berbeda dari ukuran file download.
Checkpoint refs/artifacts dipakai ulang dari cache, tetapi model objects/parameters
antar eksperimen tetap independen.

## Apa yang dimaksud training dan beberapa model sekaligus

Pilot melakukan forward, supervised patch CE, backward dan **optimizer.step**,
sehingga gradient dan Adam states benar-benar terbentuk. LoRA probe berupa adapter
linear sederhana dengan FP32 A/B dan frozen BF16 base; target q/k/v/o pada encoder dan
decoder yang benar-benar memiliki proyeksi. Tidak mengubah word/PLE tables. Initial
B=0 menjaga proyeksi base sebelum update; A dapat menerima zero gradient pada step
pertama. Ini utility profiling, bukan implementasi T02 production PEFT/checkpoint.

TRAINING_POLICY="full" opsional mengaktifkan seluruh params. AdamW moments mengikuti
dtype params aktual; tidak ada FP32 master-weight copy yang disembunyikan. Hasil full
policy tidak setara dengan recipe optimizer FP32-state/master-weight lain. Report
mencatat optimizer_tensor_dtypes dan parameter_dtypes. Jangan menganggap peak LoRA
sebagai biaya full fine-tuning.

Setelah single pilot PASS, notebook menambah satu model per tahap dan mengukur
prefix list: satu, dua, tiga, kemudian empat model **resident pada satu GPU**.
Setiap tahap menjalankan dua mode:

- **round_robin:** setiap model forward/backward bergiliran; semua weights/optimizer
  states tetap resident selama probe.
- **overlap_graphs:** forward graphs seluruh model ditahan bersama sebelum backward,
  sehingga peak mencakup gabungan activations; backward dan optimizer launches
  tetap dijadwalkan berurutan. Ini tidak mengklaim CUDA kernels berjalan paralel.

Modes memakai optimizer baru, dan moments dibersihkan setelah report. Model LoRA
weights tetap berubah oleh steps sebelumnya selama cumulative probe; memory dimensions
sama, tetapi losses bukan independent quality comparison. Tidak ada speedup claim.

Guard menggunakan **free VRAM aktual**, meta weight count setelah tying/LoRA, serta
estimated gradient + two Adam moment tensors berdasarkan trainable dtype. Saat model
berikutnya ditambahkan, calon optimizer states model resident juga dihitung. Guard
8 GiB konservatif untuk lengths/batch kecil dan bukan jaminan terhadap OOM. Untuk
lengths/batch besar, ubah headroom secara eksplisit dan gunakan measured peaks.

## Membaca hasil

JSON report menyimpan environment/config, source mapping, synthetic data policy,
loaded_memory, training snapshots, optimizer state bytes (total dan GPU), losses,
wall time serta allocator counters. Tabel ringkas menunjukkan:

- jumlah model, mode dan nama experiments;
- peak allocated/reserved GiB, optimizer bytes dan waktu;
- PASS, SKIPPED_HEADROOM atau OOM dengan stage/reason.

PASS adalah observed optimizer probe. SKIPPED_HEADROOM hanya estimator guard;
jangan menyebut jumlah tersebut "teruji tidak muat". OOM adalah kejadian aktual.
max_models_passed dihitung hanya dari successful training rows. Peak reserved dapat
memuat allocator cache; allocated/driver free/total di-report terpisah. Optimizer
bytes total dapat mencakup small CPU step tensors; optimizer_gpu_state_bytes adalah
bagian GPU saja.

Tidak ada dataset training lengkap, benchmark repair, autoregressive cache, model
publishing, RL, VM atau code execution dari dataset dalam probe ini. Hasil dua
steps tidak membuktikan convergence/repair quality. Factory M04/M05 produksi dan
S01 checkpoint tetap memiliki acceptance tambahan.

## Cell map dan dependency

0 intro → 1 control → 2 environment → 2a password input → 2b Apply/verify → 3 configs → 4 meta → 5 batch/loss →
6 merged attention → 7 model builder → 8 pretrained/profiling helpers →
9 single pilot → 10 cumulative residency → 11 JSON report; cell12 khusus pytest.

PEP723 mengikuti versi yang diuji: marimo0.25.1, Torch2.14.1, Transformers5.18.0,
safetensors0.8.0 dan pytest8.4.2, Python3.13. Hugging Face Hub disediakan dependency
Transformers; tidak ada pip upgrade di cell.

```powershell
python -m marimo check --strict notebooks/bifocalgemma_pretrained_probe.py
python -m pytest notebooks/bifocalgemma_pretrained_probe.py -q
python -m marimo edit notebooks/bifocalgemma_pretrained_probe.py
```

[Header audit](../tests/fixtures/gemma4_checkpoint_header_audit.json) memverifikasi
keempat pinned checkpoint: 540 target text state keys pada E2B, 665 pada E4B,
zero missing/shape mismatch/unexplained keys. Extra shared-KV tensors 60/54
dikenali sesuai native config. Hanya header via HTTP range yang dibaca; tensor
payload tidak diunduh ke laptop.

Local checks: strict loader fixture (missing/shape/unknown fail before copy), IT
word anchor/PLE separation, LoRA scope/zero init, real optimizer states, independent
models and two residency modes tested with native tiny tensors. Graph CPU guard and
notebook pytest inspected without checkpoint download. GPU96 execution is delegated
to the user, matching their upload workflow.


## Exact-length sweep for the larger context study

The default single pilot has a short repair target. Raising MAX_DECODER_TOKENS
alone only raises its cap; it does not test training on a longer decoder
sequence. The added sweep explicitly fills a synthetic target before the IT
closing turn so that the **shifted decoder tensor reaches the requested length**.
This artificial loss is a memory/gradient stress test, not a realistic repair
example or a quality evaluation. Real SFT examples are never padded this way.

For the sequential sweep set:

```python
RUN_PRETRAINED = True
RUN_CONCURRENCY = False
RUN_LENGTH_SWEEP = True
LENGTH_GRID = ((2048, 2048), (4096, 4096), (4096, 8192), (8192, 8192))
```

LENGTH_EXPERIMENTS defaults to the four pairs plus both IT baselines. Each pair
uses (encoder context, total paired-decoder training length). Baselines receive
the selected context in their decoder prefix and reserve encoder + paired
budget + 512 tokens, reaching 16,896 total at the 8k/8k boundary. Inactive cap
settings from the short pilot do not control this sweep.

One original model is loaded and released for each case. Cases execute
sequentially. OOM/guard failure stops larger boundaries for that model and is
recorded; the next configuration is still attempted. PASS requires finite loss,
nonzero finite gradients and AdamW updates at the **actual** tensor dimensions.
Headroom estimates do not establish an activation-memory budget.

Every attempted case is saved immediately to a timestamped length_sweep JSON
under OUTPUT_DIR, with IN_PROGRESS while running and FINISHED after the loop.
The final report also includes the whole sweep. FINISHED is a process state;
only all_configurations_passed_maximum=true shows that every requested model
passed the largest requested boundary. A failed or skipped case is not a
measured feasible training length. Partial files can be inspected after session
interruption; this does not implement optimization-state resume for probes.

The current attention implementation is eager. If the desired 8k boundary
fails, gradient/attention memory optimization is a subsequent implementation
task, followed by remeasurement. Do not infer 32k/8k feasibility from the
original architecture's 128k position limit or from inference measurements.
The sweep is prepared for the user's Molab run; no pretrained weights were
loaded locally for this update.
