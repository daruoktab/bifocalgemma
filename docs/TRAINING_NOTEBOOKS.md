# Notebook training matrix BifocalGemma

Generator [generate_full_matrix_notebooks.py](../tools/generate_full_matrix_notebooks.py)
menghasilkan sepuluh notebook standalone, masing-masing untuk satu model.
[Peta lengkap notebook](../README.md#ten-model-experiment-matrix) mencakup delapan
pasangan encoder–decoder dan baseline E2B-IT/E4B-IT decoder-only.

Tiga notebook family lama dan generator sequential telah dipensiunkan.
Dokumen [sequential training](SEQUENTIAL_TRAINING.md) serta
[perbandingan recipe](TRAINING_RECIPE_COMPARISON.md) adalah riwayat eksperimen.

## Menjalankan di Molab

1. Unggah satu notebook `train_*.py` dari matriks.
2. Jalankan setup; loader memverifikasi ZIP runtime yang dipin dan dependency.
3. Sediakan `HF_TOKEN` atau isi form token, lalu tekan **Jalankan**.
4. Pipeline menjalankan training/resume, export/upload, dan sepuluh benchmark.
5. Periksa dashboard, status sesi, serta receipt upload; status tertunda belum
   membuktikan artefak remote lengkap.

Token diperlukan untuk akses dataset/model dan upload; notebook memakai
repository model yang ditentukan untuk konfigurasi tersebut. Tidak diperlukan
checkout repo atau instalasi `bifocalgemma` dari PyPI.

## Cell dan data contract

Cell menyajikan pengantar, aktivasi runtime, form token/model, monitor live,
dan pemanggilan `run_matrix_training_and_benchmark`.
Source utama berada di modul canonical, bukan disalin sebagai training loop
di setiap notebook.

Dataset dibaca dari koleksi beku `mixed_context_enc32k_dec8k_v4` pada revision
immutable `4b8d6b366d68c4c89feaf450407ebe2b51e87379` (tercantum di
[config.py](../src/bifocalgemma/config.py)). Konfigurasi V4 menggunakan 47.932
contoh training dan 7.098 contoh dev, `1.5` epoch, `batch_size=2`,
`accumulation=8` (`effective_batch=16`), menghasilkan total 4.494 optimizer steps
dengan dynamic length-grouped micro-batching. Schedule dibuat sekali dan
dipakai sesuai identitas run; pipeline utama tidak mengganti contoh dengan
comprehension replay.

Delapan model paired BifocalGemma menjalankan Stage 1 frozen-decoder alignment
20% (899 steps, `lambda_ptr=0.10`) diikuti Stage 2 joint SFT 80% (3.595 steps,
`lambda_ptr=0.03`). Baseline decoder-only (`e2b-it`, `e4b-it`) menjalankan
1-stage Full Decoder SFT selama 4.494 steps. Optimizer/scheduler meneruskan
state sesuai kontrak transisi tahap.

## Evaluasi dan resume

Sepuluh suite benchmark V4 terdiri atas 8 suite penuh
(`longcodeqa_gemma32k_dec8k_v1`, `reponiah_gemma32k_dec8k_v1` 4K–32K,
`humanevalfix_python_full_v1`, `humanevalfix_python_v2`,
`sweqa_oracle_gemma32k_dec8k_v1`, `sweqa_noisy_oracle_gemma32k_dec8k_v1`,
`gsm8k_zero_shot_v2`, `gsm8k_eight_shot_v2`) dan 2 suite deterministik
500 sampel (`crosscodelong_chunk_gemma32k_dec8k_v1`,
`crosscodelong_function_gemma32k_dec8k_v1`).

Reuse benchmark mengikat identitas settings dan checksum checkpoint.
Export lokal harus tervalidasi; jika upload aktif, metadata konten remote juga
harus cocok sebelum completed-run reuse. `PENDING_UPLOAD` tetap menandai
sinkronisasi yang belum selesai meskipun evaluasi sudah ada di cache.

Kode bersih menolak checkpoint/schedule sebelum refactor. Agar run lama di
Molab tetap berjalan dan bisa restart, simpan notebook, runtime, dan artefak lama.
Jangan mengganti source notebook aktif dengan versi baru atau memakai direktori
run lama untuk eksperimen baru.

## Regenerasi dan rilis

```powershell
python tools/generate_full_matrix_notebooks.py
python tools/generate_full_matrix_notebooks.py --check
Get-ChildItem notebooks/train_*.py | ForEach-Object { python -m marimo check --strict $_.FullName }
```

Regenerasi lokal membuat ZIP dan `LOCAL-PREVIEW` jika source hash berubah.
Notebook preview memerlukan ZIP lokal yang cocok dan belum siap dibagikan
sebagai file standalone. Publikasi, verifikasi anonymous read-back, dan
`--check --require-published` dijelaskan di [RUNTIME_RELEASE.md](RUNTIME_RELEASE.md).
Publikasi hanya menambah ZIP immutable dan tidak mengganti `runtime/latest.zip`.

## Lokasi run matrix baru

Konfigurasi matrix V4 memisahkan lokasi paired BifocalGemma (8 model) pada subfolder HF `sliding-v4/` dan root lokal `outputs/matrix/sliding-bifocal-v4/`, sedangkan baseline decoder-only (`e2b-it`, `e4b-it`) memakai subfolder HF `latest/` dan root lokal `outputs/matrix/versionless-repair-v4/`. Hasil lama disimpan di folder `archive/YYYY-MM-DD/` pada masing-masing repo model. Resume mencari checkpoint di lokasi yang sesuai dengan konfigurasi run. Run V3 lama memerlukan notebook, ZIP dan konfigurasi asli serta lokasi checkpoint yang sesuai. Untuk E4B V3 di step 7.462, notebook yang menunjuk ZIP kompatibel ada pada [commit 071b331](https://github.com/daruoktab/bifocalgemma/blob/071b331/notebooks/train_e4b_it.py).

Runtime baru mempertahankan whitespace target unified diff saat menyusun label training. `learnable_cross_scale=False` membekukan skala cross-attention pada inisialisasi dan di kedua tahap training. Hash semantik run baru berbeda dari V3 dan dari rilis versionless sebelumnya.

Kontrol offline memakai 44.408 target training dan 4.934 target dev dari salinan pool proyek, dua tokenizer decoder lokal, dan prompt lengkap pada sampel dekat batas token. Pada evaluasi, seluruh patch referensi HumanEvalFix lulus tes (164/164). Jawaban referensi juga lulus scorer untuk 107 LongCodeQA, masing-masing 9.072 SWE-QA oracle/noisy, masing-masing 1.319 GSM8K zero/eight-shot, dan dua sampel tetap CrossCodeLong berisi 100 kasus per jenis. Kontrol ini memakai jawaban referensi sebagai input scorer. Kualitas keluaran dari checkpoint terlatih masih perlu diukur pada run matrix baru.
