# Validasi Scaffold dan Native Tiny

Diperiksa **6 Oktober 2026**: Conda ai, Python 3.13.16, Torch 2.14.1+cu130,
Transformers 5.18.0, CUDA 13.0, NVIDIA GeForce RTX 3060 Laptop GPU (6144 MiB).
Tidak ada pretrained model weights yang diunduh atau dimuat.

## Hasil aktual

| Check | Hasil | Cakupan |
| --- | --- | --- |
| Unit suite | **55 tests lulus, tanpa skip** | 41 existing + 8 native integration + 6 offline data/tokenizer |
| Native integration | 8 tests lulus | Six-model backward, no-context parity, masks, KV ownership, aliases |
| Offline data/tokenizer | 6 tests lulus | Cached real tokenizers, budgets/hashes, synthetic patch roundtrip |
| CLI smoke-native CPU | Enam model lulus | Full-depth random native forward/backward |
| CLI smoke-native CUDA | Enam model lulus | Finite/nonzero gradients, aliases after GPU transfer, measured tiny memory |
| Ruff lint/format dan ty | Lulus | Source/tests dengan environment ai |
| Offline wheel | Lulus | Config/hash assets dan modul native/data terdapat dalam wheel; tanpa dependency download |
| Docs/block map | Lulus | 18 Markdown files, 23 source blocks/anchors, relative file links |
| Existing checks | Tetap dalam suite | Meta shapes, reference/bridge, data collation/loss, reporting, registry |

Dua tes tokenizer riil memakai BIFOCALGEMMA_TOKENIZER_CACHE menuju cache audit.
Tanpa environment variable ini, kedua tes tersebut di-skip; tests lainnya tetap
portable/offline. Output tabel pada test_evaluation memakai angka fixture untuk
menguji rendering, bukan hasil benchmark BifocalGemma.

## Native tiny CUDA smoke

Batch synthetic dua contoh; logits `[2,5,128]`; FP32, eager, tanpa optimizer/cache.
Parameter unik setelah alias dihitung dari model yang benar-benar dibuat.

| Model | Parameters tiny | Peak allocated bytes | Peak reserved bytes |
| --- | ---: | ---: | ---: |
| E2B–E2B | 1,793,224 | 31,927,808 | 37,748,736 |
| E4B–E4B | 3,129,592 | 42,767,360 | 48,234,496 |
| E2B–E4B | 2,468,064 | 37,413,376 | 48,234,496 |
| E4B–E2B | 2,468,064 | 37,413,376 | 48,234,496 |
| E2B-IT baseline | 898,660 | 24,565,760 | 48,234,496 |
| E4B-IT baseline | 1,567,868 | 30,006,272 | 48,234,496 |

Peak counters di-reset per model dalam satu process; reserved bytes dapat membawa
allocator cache model sebelumnya. Ini bukan total GPU process memory atau profile
pretrained. CPU melaporkan parameter/gradient bytes dan null untuk CUDA peaks.
Reference generic M03 tetap terpisah (sekitar 21k–45k parameter).

## Perilaku yang dibuktikan

- Keempat pasangan dan dua baseline melakukan native forward/backward dengan finite
  logits/gradients. Ties hanya pada pasangan simetris; decoder head selalu tied.
- Decoder tanpa context cocok dengan native Gemma pada token valid, termasuk left
  padding dan panjang 513 yang melampaui local window. Final softcap dipertahankan.
- Token decoder masa depan tidak memengaruhi output sebelumnya. Context encoder
  berpengaruh, bidirectional encoder membaca suffix, dan padded context setara.
- Entirely masked encoder/query rows menghasilkan finite backward. Local self
  window tidak menyembunyikan encoder keys.
- Shared layers menggunakan KV pemilik bertipe sama dan tidak memiliki k_proj/v_proj
  tambahan. Native PLE, MLP, residual dan RoPE tetap berada dalam upstream stack.
- Ties/PLE separation serta logits dipertahankan pada perubahan dtype dan state_dict
  reload (termasuk assign=True). CPU→CUDA alias diperiksa melalui smoke output.
- Tokenizer keenam konfigurasi menjaga Unicode/tab/indentasi, BOS sekali, closing
  turn token, patch-only target, zero prefix labels dan single shift. Cache hilang,
  hash/revision salah, target kosong dan overflow ditolak.
- D06 membalik bug-inducing patch SWE-smith pada synthetic single/multi-file fixtures,
  memverifikasi apply/roundtrip di temporary directory, serta menolak ref/hash/patch
  salah. Tidak ada source checkout yang dimodifikasi oleh normalizer.

## Batas bukti

M07 menguji native random-weight integration dengan width/vocabulary diperkecil.
M04 pretrained transfer dan M05 production factory masih placeholder. D03 tersedia,
D06 local normalization tersedia, tetapi D04 download/reconstruction/overlap audit
real dataset belum siap. Tidak ada official SWE-smith/SWE-bench harness execution,
repair score, real SFT, optimizer resume, autoregressive merged cache, optimized
kernel, gradient-checkpointing integration, BF16/quantized merged validation, atau
peak VRAM model penuh. Direct SFT masih hipotesis yang memerlukan pilot pretrained.

## Reproduksi

```powershell
conda activate ai
$env:PYTHONPATH = 'src'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
$env:BIFOCALGEMMA_TOKENIZER_CACHE = 'C:\path\to\gemma4-json-audit'
python -m unittest discover -s tests
python -m bifocalgemma validate-config
python -m bifocalgemma smoke-native --experiment all --device cpu
python -m bifocalgemma smoke-native --experiment all --device cuda
ruff check .
ruff format --check .
ty check .
git diff --check
```

Isi cache root dengan path audit milik pengguna; bukan folder tokenizer HF umum.
[Native local validation](NATIVE_LOCAL_VALIDATION.md) menjelaskan API, assets,
kontrak data dan sumber riset. [Build guide](BUILD_GUIDE.md) menyatakan status per blok.

## Notebook marimo standalone — 6 Oktober 2026

[Notebook](../notebooks/bifocalgemma_architecture_test.py) berisi 13 cell dan PEP723
minimal. Marimo check --strict, Ruff lint/format dan ty lulus. Cell pytest-only
memuat tiga tes yang lulus lewat python -m pytest notebook: mask local/global,
prefix/padding loss gradients, serta finite fully-masked backward. Graph CPU FP32
berhasil, serta marimo HTML export pada CUDA dengan BF16 forward/backward dan
FP32 semantic parity/causality checks berhasil: enam model, 60 kelompok checks.
Reference decoder menyalin nonpersistent buffers sebelum mengganti attention;
semantic FP32 tolerance tetap 2e-5. Ini tidak mengklaim BF16 logits parity.
Full-size random BF16 probe untuk GPU 96 GB tersedia dengan RUN_FULL_SIZE=False
secara default; belum dieksekusi di GPU Blackwell dari sesi lokal ini.
Report JSON dan HTML berada di outputs yang ignored. Tidak ada download bobot/data.
Cara memakai dan batas probe ada pada [panduan marimo](MARIMO_ARCHITECTURE_TEST.md).

## Pretrained memory probe — notebook siap Molab, belum hasil GPU96

[Notebook standalone](../notebooks/bifocalgemma_pretrained_probe.py) dan
[panduan](PRETRAINED_GPU_PROBE.md) menyediakan pinned text checkpoint transfer,
attention LoRA/full policy, AdamW steps, serta cumulative 1–4 model residency.
CPU preflight menampilkan SKIPPED_HEADROOM sebelum network/model weights download.
Lima helper tests memakai native tiny safetensors fixture, termasuk atomic key/shape
validation, shared IT word anchor, PLE separate, LoRA scope dan actual optimizer
state formation. Tiga notebook pytest tests lulus. Marimo check --strict, Ruff dan
ty diperiksa. Pengguna mengunggah dan menjalankan probe asli di Molab; checkpoint
asli tidak dimuat lokal dan belum ada angka peak VRAM/kapasitas GPU96 yang diukur.

Header safetensors asli keempat checkpoints diperiksa via HTTP range terhadap
state_dict native meta: E2B 540 target keys / 60 ignored shared-KV, E4B 665 / 54;
zero missing, shape mismatch atau unexplained text key. Audit kecil tersimpan di
[fixture header](../tests/fixtures/gemma4_checkpoint_header_audit.json).
Payload tensor tetap belum diunduh atau diuji lokal.
