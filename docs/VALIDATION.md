# Validasi Scaffold Saat Ini

Pembaruan construction scaffold: **6 Oktober 2026**.
Environment yang dipakai: Conda ai, Python 3.13.16, Torch 2.14.1+cu130,
Transformers 5.18.0, CUDA 13.0, NVIDIA GeForce RTX 3060 Laptop GPU.

## Checks yang dijalankan

| Check | Hasil | Cakupan |
| --- | --- | --- |
| Unit suite | 41 tests lulus, tanpa skip | Existing native/meta/reference tests + 15 construction tests |
| Construction tests, termasuk batch baseline | 15 tests lulus dalam suite akhir | Ties/gradients, masks, collation, loss, wire format, registry, fail-fast trainer |
| Ruff lint dan format | Lulus | Python source/tests |
| ty | Lulus | Source/tests memakai environment ai |
| CLI validate-config | Lulus | Enam canonical IDs, revisions, direct_sft, no mandatory teacher |
| CLI smoke CPU | Empat pasangan lulus | Tiny random-weight forward/backward, finite loss/gradients |
| CLI smoke CUDA | Empat pasangan lulus | Tiny reference dan sharing policy pada GPU |
| CLI inspect | Lulus | Native profile widths, base/IT provenance, baseline no encoder |
| Documentation links/block map | Diperiksa | Relative paths, source slots dan build-guide anchors |
| git diff whitespace | Diperiksa | Perubahan yang disiapkan |

Contoh smoke memiliki logits shape [2,5,128], reference widths 32/48 dan sekitar
21k–45k parameter. Angka itu milik toy assembly, bukan estimasi full Gemma.
Synthetic loss tidak dilaporkan sebagai kualitas model atau benchmark.

## Perilaku penting yang diverifikasi

- Cross-component symmetric word aliases memakai decoder anchor, preserve input
  table values saat tying, dan menerima gradient encoder+decoder pada satu Parameter.
- Asimetris mempertahankan encoder table terpisah; decoder head tetap alias decoder words.
- Reference attention memakai joint competition self/encoder keys, bukan dua softmax
  yang dijumlahkan. Decoder future tokens tidak memengaruhi prefix output; encoder
  context dapat memengaruhi output.
- Local self window tidak memangkas encoder context. Entirely masked query rows
  menghasilkan zero output dan finite gradients.
- Baseline batches menerima encoder arrays kosong; batch tidak mencampur model
  decoder-only dan encoder-decoder.
- Prefix/target di-shift tepat sekali; prefix/pad tidak memperoleh CE gradient.
  All-ignored loss finite/differentiable zero.
- Encoder formatter tidak mengambil field target_patch. Prediction JSONL tidak
  membawa target/test artifacts dan memakai tiga key SWE-bench yang benar.
- Default spec/config tidak meminta distillation/PLe sharing. Placeholder trainer
  fail dengan T02/docs link, tanpa memuat dataset atau bobot.

## Bukti native yang berbeda dari reference

test_gemma4_topology memeriksa full-size text parameter shapes pada meta,
35/42 layer schedules, PLE, MLP dan KV-sharing. Thin native Gemma4TextModel
tests memakai random weights serta forward/backward/cached parity.
test_bridge memeriksa separated bridge, termasuk CUDA.

Tes itu tidak membuktikan native merged decoder, M04 weight transfer atau M05 model
assembly sudah bekerja. ReferenceMergedAttention belum memiliki native RoPE,
inter-layer KV sharing, PLE dan cache. Factory, tokenization/data adapters,
trainer, generation, agent, real harness serta checkpoint masih placeholder.

## Yang belum diuji sebagai experiment

Tidak ada download bobot pretrained, real dataset training, optimization run,
BifocalGemma checkpoint, code-repair score, SWE-bench Docker evaluation, long-context
native merged cache, BF16/quantized merged kernel, convergence atau measured full-model
VRAM. Run-ready status tidak diperoleh hanya dari printed tables atau metadata.

## Reproduksi

```powershell
conda activate ai
$env:PYTHONPATH = 'src'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
python -m unittest discover -s tests
python -m bifocalgemma validate-config
python -m bifocalgemma smoke --experiment all --device cpu
python -m bifocalgemma smoke --experiment all --device cuda
ruff check .
ruff format --check .
ty check .
git diff --check
```

CUDA command hanya saat perangkat tersedia. Full production commands remain
documented boundaries; lihat [build guide](BUILD_GUIDE.md). Hasil lebih lanjut
harus mencatat commit/config/data/hardware/protocol baru, tidak menimpa angka
historis dengan klaim yang belum dijalankan.
