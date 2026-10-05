# Panduan Tim: Membangun BifocalGemma Per Blok

## Tujuan bersama

Bangun dan ukur direct-SFT merged encoder base / decoder IT untuk empat pasangan,
dengan dua baseline IT. Symmetric ordinary words tied sejak initialization; PLE
terpisah. Distillation/conversion tidak wajib. Ketentuan kompetisi mengikuti
[status verifikasi](COMPETITION_GUIDE.md), bukan deadline yang diasumsikan.

| Anggota | Area kerja |
| --- | --- |
| daruoktab | M04/M05 native initialization/integration, M06 cache, T03 profile, T02 optimizer/PEFT, S01 checkpoint |
| attmhd | D03/D04 data/tokenization, D05 AST bila diperlukan, E01 harness, I01/A01 evaluation plumbing, analysis/paper |

Pembagian ini dapat disesuaikan, tetapi interface di contracts/config dibahas bersama.
[Build guide](BUILD_GUIDE.md) menentukan dependency dan acceptance criteria.

## Milestone berbasis bukti

| Milestone | Hasil yang diperlukan |
| --- | --- |
| 1. Reference blocks | Registry, collator, tying, joint softmax/GQA, loss dan smoke/tests |
| 2. Native tiny integration | M07 native layers, positions, PLE dan shared KV; M04/M05 pretrained tetap terpisah |
| 3. Data/profile | Pinned real records/tokenizers, overlap audit dan actual memory/latency |
| 4. Training/checkpoint | Direct SFT/PEFT budget tercatat, alias-safe resume, loss/retention diperiksa |
| 5. Inference/harness | Patch generation, tools bila digunakan, valid wire format dan real test grading |
| 6. Four-pair study | Same tasks/protocol, asymmetry comparison, tying controls bila ada, uncertainty dan failure analysis |

Milestone 1 dan native tiny milestone 2 tersedia; D03 offline dan D06 patch contract
juga dapat dijalankan. Milestone 3 masih memerlukan data riil dan profiling model
penuh. Kelulusan native tiny tidak menyelesaikan transfer pretrained atau training.

## Workflow lokal

```powershell
conda activate ai
$env:PYTHONPATH = 'src'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
python -m bifocalgemma validate-config
python -m bifocalgemma smoke --experiment all --device cpu
python -m unittest discover -s tests
ruff check .
ruff format --check .
ty check .
git diff --check
```

Editable installation opsional jika semua dependency tersedia. Commands reference
tidak memerlukan bobot pretrained atau download dataset. Checks lulus hanya memberi
bukti pada kontrak yang diperiksa; tidak menjamin quality/fit pada model penuh.

## Mengubah satu blok

1. Baca bagian blok, source dan dependency di BUILD_GUIDE.
2. Pertahankan typed I/O dan beri hasil error eksplisit jika dependency belum siap.
3. Implementasikan acceptance criteria dengan test yang mengukur perilaku, bukan
   hanya memanggil method atau menyalin rumus implementation.
4. Perbarui status di scaffold.py, docs dan VALIDATION dengan cakupan aktual.
5. Jika kontrak berubah, perbarui caller/config/checkpoint metadata pada commit yang sama.

Belum ada bobot besar yang dipindahkan ke repo. Dataset/checkpoint/output tetap
ignored; configs, fixtures kecil dan protocol manifests yang aman dapat dilacak.
Jangan memakai angka dummy sebagai result table.

## Data dan eksperimen

Baseline mendapat data/context/tool/generation budget sebanding. Record precision,
trainable params, seed, source/data revisions, selection policy, encoder/decoder
lengths dan cache modes. Sumber/pemisahan ada dalam DATASETS_AND_BENCHMARKS dan
DATA_PIPELINE. E2B/E4B juga berbeda layout; hasil tidak digeneralisasi ke semua LLM.

Dokumen aktif harus menjelaskan satu recipe. Rationale/perubahan historis ditaruh
di RESEARCH_LOG, bukan membiarkan instruksi default yang saling bertentangan.

## Git review

```powershell
git pull --ff-only origin main
git switch -c codex/nama-blok
git add <file-yang-diubah>
git commit -m "feat: implementasikan blok dengan kontrak dan evidence"
git push -u origin HEAD
```

Tinjau diff dan pastikan output besar/secrets tidak ikut staged. Usulkan perubahan
yang konkret beserta validation evidence; status ready hanya setelah dependency
dan acceptance benar-benar terpenuhi.
