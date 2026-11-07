# Native Local Validation: Langkah Sesudah Reference

Implementasi **6 Oktober 2026** menggunakan Transformers **5.18.0**, Conda `ai`,
dan bobot acak. Ini integrasi native Gemma pada ukuran tensor kecil. Ia tidak
memuat checkpoint base/IT, melakukan SFT pretrained, atau menghasilkan skor repair.
Hasil aktual berada di [VALIDATION](VALIDATION.md).

## Temuan riset dan keputusan

1. [T5Gemma 2](https://arxiv.org/html/2512.14856v2) melakukan adaptasi UL2 sekitar
   2T token sebelum post-training. Direct patch SFT dari encoder base/decoder IT
   tetap hipotesis BifocalGemma. Kelulusan tes arsitektur tidak membuktikan recipe
   tersebut dapat menggantikan adaptasi pretraining; pilot pretrained diperlukan.
2. [Gemma 4 native](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/gemma4/modeling_gemma4.py)
   menghilangkan proyeksi K/V pada shared layers. Merged integration mempertahankan
   pemilik KV dan berbagi hasil proyeksi self serta encoder pada forward yang sama.
3. [Harness SWE-smith](https://raw.githubusercontent.com/SWE-bench/SWE-smith/main/swesmith/harness/utils.py)
   membalik gold bug patch untuk memperbaiki snapshot. Referensi instance buggy
   berbeda dari asumsi base_commit SWE-bench; evaluator juga menangani test-file
   removal commit. Snapshot input model tidak boleh dibangun dari checkout evaluator
   yang sudah mengembalikan tes tersembunyi.
4. Dari config audit, dua packed PLE dan satu ordinary word table bersama saja
   membutuhkan **9,5 GiB (E2B–E2B)** atau **11,75 GiB (E4B–E4B)** dalam BF16.
   Rumus bytes: `2 * vocab * layers * ple_width * 2 + vocab * hidden_width * 2`.
   Angka ini adalah storage matematis tabel, bukan peak VRAM. GPU lokal 6 GiB;
   [bitsandbytes linear quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
   tidak otomatis mengkuantisasi embedding. Frozen/offloaded PLE adalah pekerjaan
   lanjutan yang perlu profiling, bukan janji bahwa QLoRA langsung muat.

## M07 — factory native tiny

API [build_native_tiny_model(spec)](../src/bifocalgemma/models/v2/native.py) terpisah
secara eksplisit dari M04/M05 pretrained factory. Semua bobot diinisialisasi acak.
`native_config(size, tiny=False)` hanya untuk inspeksi bentuk pada `meta`.
Metadata kecil dikemas dalam package assets sehingga factory tidak bergantung
pada lokasi folder tests atau working directory.

| Properti | E2B tiny | E4B tiny |
| --- | --- | --- |
| Hidden width | 32 | 48 |
| Layers | 35 | 42 |
| Vocabulary / PLE vocabulary | 128 | 128 |
| PLE width | 4 | 4 |
| Query / KV heads | 8 / 1 | 8 / 2 |
| Local / global head dimension | 8 / 16 | 8 / 16 |
| Shared-KV layers | 20 | 18 |
| Local window | 512 | 512 |

Layer schedule, MLP variant, PLE, embedding scaling, native norm/residual, RoPE
termasuk partial global rotation, value normalization, dan final logit softcap
mengikuti config/native implementation. Encoder memakai bidirectional attention
native dengan local window-nya. Decoder baseline tetap `Gemma4TextModel` causal.

Pasangan mengganti hanya attention decoder. Q dan self K memakai native RoPE;
cross K memakai native K norm tanpa decoder RoPE. Self/cross memakai proyeksi
K/V yang sama dan satu softmax; window local hanya membatasi self keys. Shared
layers menggunakan concatenated K/V pemilik terakhir bertipe sama. Tidak ada
proyeksi K/V baru pada shared layers.

Satu `encoder_projection` di tingkat model mengubah Henc menjadi Ddec untuk
asimetris; simetris memakai Identity. `encode_context` pada factory tiny mengembalikan
**H yang sudah disejajarkan**, lalu `decode_logits` menerima H tersebut. Ordinary
words simetris mengikuti alias decoder; PLE tidak pernah di-alias. Alias dipasang
ulang setelah `.to(...)`/perubahan dtype dan `load_state_dict`, termasuk assign=True.
Ini memverifikasi state_dict tiny, bukan format production checkpoint S01.

Eager attention, float32 default, `use_cache=False`; permintaan temporal cache
ke merged attention ditolak. Belum ada optimized kernel, pretrained transfer,
quantization, gradient-checkpointing integration, atau autoregressive generation.

```powershell
conda activate ai
$env:PYTHONPATH = 'src'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
python -m bifocalgemma smoke-native --experiment all --device cpu
python -m bifocalgemma smoke-native --experiment all --device cuda
```

CLI mencetak satu JSON per model: dimensi/depth, unique parameters, synthetic loss,
finite/nonzero gradients, parameter/gradient bytes, dan peak allocated/reserved
CUDA. CPU peak bernilai null; tensor bytes bukan process RAM peak. Pengukuran GPU
adalah native tiny pada panjang synthetic `[2,5,128]` logits, bukan model penuh.
CLI `smoke` lama tetap menguji reference generic; `smoke-native` menguji M07.

## D03 — tokenizer offline

[tokenize_repair_example](../src/bifocalgemma/data/tokenization.py) menerima
`example, spec, tokenizer_cache`, serta keyword limits `max_encoder_tokens=4096`
dan `max_decoder_tokens=4096`. Limit harus 1..131072; overflow menimbulkan ValueError.
SFT membutuhkan target nonempty; API target-free generation tetap I01.

Cache mempunyai struktur `raw/google__gemma-4-E2B/`, `raw/google__gemma-4-E2B-it/`
dan padanannya E4B. Isi berasal dari audit sebelumnya: lima JSON dan IT Jinja.
Loader memeriksa revision registry terhadap manifest package dan SHA-256 setiap
asset sebelum memanggil AutoTokenizer dengan local_files_only=True. Tidak ada
fallback network. Cache root diberikan caller, tidak di-hardcode ke komputer lama.

Base BOS dipasang sekali oleh tokenizer. IT memakai template resmi dengan
`enable_thinking=False`, tanpa tools; target diakhiri `<turn|>`, tanpa melatih newline
setelah stop token. Full conversation harus mempertahankan prefix token persis;
bila BPE melintasi batas prefix/target, contoh ditolak secara eksplisit.

Context adalah issue + seluruh file, diurutkan path. Pasangan menerima context
pada encoder dan instruksi/issue di prefix IT; baseline menerima context yang sama
pada prefix IT. Tidak ada retrieval/truncation berbasis gold. Metadata output
merekam hash/revision, policy, token counts, boundaries dan limits. Offset AST
belum diekstrak. IDs asli tidak boleh dimasukkan ke tiny vocabulary 128; dua jalur
tes terpisah dan tidak melakukan clipping/modulo ID.

```powershell
# Root cache audit lokal milik pengguna; tidak dimasukkan ke Git.
$env:BIFOCALGEMMA_TOKENIZER_CACHE = 'C:\path\to\gemma4-json-audit'
python -m unittest tests.test_offline_data -v
```

Dua tes tokenizer riil di-skip bila environment variable tidak disetel. Jika disetel
ke cache salah/rusak, tes gagal; tidak di-skip atau mengunduh pengganti.

## D06 — kontrak lokal SWE-smith

[SWESmithRecord](../src/bifocalgemma/data/swesmith.py) menyimpan dataset_revision,
buggy_ref, buggy_snapshot_sha256, snapshot, issue, dan bug_patch. Pada kontrak ini
`snapshot.base_commit` menunjuk **buggy_ref eksplisit**, bukan field upstream yang
diasumsikan tersedia. Revision upstream harus dipin saat D04 mengambil data riil.

`normalize_swesmith_record(record, split=...)` mengembalikan RepairExample dengan
target repair serta provenance: hash buggy/fixed snapshot dan bug/repair patch.
Hash snapshot memakai JSON UTF-8 daftar `(path,content)` terurut dengan separator
kompak; duplicate paths ditolak. Semua penerapan patch berlangsung di temporary
directory, bukan source checkout. Git memeriksa reverse patch dan round-trip
buggy → fixed → buggy. Kode dataset tidak dieksekusi.

Normalizer lokal sengaja menerima hanya text modifications pada existing files,
path tanpa quoting/whitespace/hidden metadata, dan konten UTF-8 berakhiran newline.
Add/delete/rename/binary/mode changes ditolak. Input snapshot/reference/hash salah
atau patch yang tidak berlaku juga ditolak. D04 production adapter tetap placeholder.

[Fixture single/multi-file](../tests/fixtures/swesmith_local.synthetic.json) sepenuhnya
sintetis, dengan revision synthetic-schema-v1. Validation label selalu
`patch_roundtrip_only_not_official_tests`; keberhasilan Git apply bukan bukti
FAIL_TO_PASS/PASS_TO_PASS lulus. Download/reconstruction instance SWE-smith asli,
license/overlap audit, dan official harness masih pekerjaan berikutnya.

## Sesudah milestone ini

Lanjutkan M04 transfer bobot dan M05 assembly produksi pada hardware yang sesuai,
D04 paired-code EditPackFT adapter/manifest, T03 profiling, kemudian pilot direct
SFT E2B–E2B. Recipe utama memakai file lokal tanpa RL atau VM/Docker; SWE-smith
environment/harness ditunda. Detail ada pada DATASETS_AND_BENCHMARKS.
Sebelum memperluas ke empat pasangan, ukur kemampuan memakai context dan retensi
format IT pada held-out data. Model tiny tidak menjawab apakah direct SFT cukup.

## Probe asli pada Molab

[Pretrained GPU probe](PRETRAINED_GPU_PROBE.md) tersedia sebagai notebook standalone
terpisah. Ia memuat pinned base/IT text weights, melakukan attention-LoRA/AdamW
microsteps, dan membandingkan 1–4 independent resident models. Loader/optimizer
helpers diuji pada tiny fixtures; GPU96/pretrained measurements dijalankan pengguna.
Ini prototype profiling, bukan penyelesaian production trainer/cache/checkpoint.
