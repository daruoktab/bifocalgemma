# Pipeline Data: Snapshot → Prefix/Target → Batch

## Kontrak antar blok

[contracts.py](../src/bifocalgemma/contracts.py) mendefinisikan:
RepositoryFile(path,content), RepositorySnapshot(repo,base_commit,files),
RepairExample(instance_id,source,split,snapshot,issue,target_patch),
TokenizedRepairExample(encoder_ids,decoder_prefix_ids,target_ids), dan Seq2SeqBatch.

Path snapshot adalah POSIX relatif; path absolut, drive dan parent traversal
ditolak. Content merupakan versi sebelum fix. target_patch hanya untuk supervisi
train/dev; input inference tidak memerlukan target. test_patch, FAIL_TO_PASS dan
PASS_TO_PASS merupakan artifacts evaluator dan tidak masuk kontrak input model.

[Contoh synthetic](../examples/repair_record.synthetic.json) menjelaskan bentuk record.
Ia bukan data SWE-smith atau contoh benchmark terverifikasi.

## D04 — sumber data dan manifest

[Dataset adapters](../src/bifocalgemma/data/adapters.py) menyediakan slot
EditPackFTAdapter, CommitPackFTAdapter, SWESmithAdapter dan SWEGymAdapter. iter_examples menerima
manifest sumber lokal/revision dan split, lalu menghasilkan RepairExample.
Fetching, checkout snapshot, filtering dan overlap audit belum diimplementasikan.
D06 [normalizer lokal](../src/bifocalgemma/data/swesmith.py) memeriksa snapshot hash
dan membalik bug-inducing patch SWE-smith menjadi repair. Referensi snapshot buggy
harus eksplisit; lihat [kontrak lokal](NATIVE_LOCAL_VALIDATION.md#d06--kontrak-lokal-swe-smith).

Manifest menyimpan source/revision, instance ID, repo/base_commit, license,
file selection, token lengths, split, filter reason serta hash patch/issue.
Pisahkan train/dev per kelompok task/repo agar trajectory atau commit terkait
tidak bocor. Audit overlap issue/PR/commit/patch/near-duplicate dengan evaluation.
Jumlah baris setelah filter lebih penting daripada jumlah pada card yang berubah.

Data utama pilot adalah EditPackFT; CommitPackFT Python tambahan opsional setelah
dedup. Adapter menggunakan instruction/subject dan old_contents sebagai input,
serta unified diff before→after sebagai target. Field content/ndiff tidak menjadi
input karena dapat membawa jawaban. SFT/data preparation tidak menjalankan kode
repository, RL, VM atau Docker. SWE-smith/Gym dan trajectories ditunda dari jalur
utama. Single-file pilot bukan bukti kemampuan lintas file. Detail ada pada
[riset dataset](DATASETS_AND_BENCHMARKS.md).

## D03 — tokenization

format_encoder_context tersedia: issue + file sebelum fix, diurutkan menurut path.
Fungsi ini tidak mengambil target_patch. Tokenization offline tersedia melalui D03:

1. Base tokenizer memproses context dan menambahkan BOS tepat sekali.
2. IT tokenizer memakai chat_template.jinja yang dipin untuk prefix instruksi/issue.
   Prefix belum memuat jawaban; enable_thinking=false dan tanpa tools.
3. Target patch menjadi continuation assistant, dengan closing token yang sesuai.
4. Context memakai seluruh file terurut pada keenam model. Batas encoder/decoder
   default 4096, maksimum 131072; overflow ditolak tanpa truncation. Limit 1e30
   pada tokenizer bukan batas eksperimen.
5. Metadata menyimpan hash/revision, policy, prefix/target boundaries, token lengths
   dan truncated=false. Original offsets untuk AST masih D05.

Untuk baseline decoder-only, D03 meletakkan context file/issue yang sama di dalam
IT decoder prefix dan menghasilkan encoder_ids kosong. Collator menerima batch
baseline [B,0] pada sisi encoder; satu batch tidak mencampur baseline dan pasangan.
Native baseline forward tidak memanggil encoder, H adapter atau merged branch.

Tokenizer raw IDs kompatibel base/IT, tetapi BOS post-processing berbeda.
Do not encode a rendered IT template with an extra BOS. Encoder visibility
tidak membuat gold patch boleh dimasukkan ke context. [Audit](GEMMA4_JSON_AUDIT.md)
menjelaskan EOS, normalizer BPE dan grammar tool.

## D02 — collator yang sudah tersedia

[RepairCollator](../src/bifocalgemma/data/collator.py) menerima record yang sudah
ditokenisasi, melakukan left padding, serta mempersiapkan teacher forcing:

```text
prefix = [2,105,7]            # synthetic IDs, bukan contoh prompt lengkap
target = [18,19,106]
decoder_input = [2,105,7,18,19]
labels = [-100,-100,18,19,106]
```

Pada posisi terakhir prefix, model memprediksi token target pertama. Prefix
dan pad tidak memperoleh CE loss. Target sendiri menjadi riwayat kausal untuk
memprediksi token sesudahnya; ini teacher forcing, bukan leakage ke encoder.
Loss T01 memakai labels yang sudah aligned dan tidak men-shift ulang.

Batch menyediakan encoder/decoder IDs Long, validity masks Bool, labels Long
dengan ignore_index=-100, dan position IDs dari cumulative validity mask.
Semua field dapat dipindahkan melalui batch.to(device). Tidak ada teacher
network pada teacher forcing; istilah itu bukan knowledge distillation.

## D05 — fitur AST opsional

[AST extraction](../src/bifocalgemma/data/ast.py) masih placeholder; existing
ASTGraphProjector hanya meng-embed tipe node/kedalaman yang diberikan.
Tree-sitter memberi byte offsets; tokenizer biasanya memberi character offsets.
Alignment harus memakai teks original, termasuk Unicode, spasi, special tokens
dan pad. Jumlah features harus sama dengan encoder sequence.

Graph edges/call resolution/GNN tidak tersedia. AST tidak masuk recipe default
sampai extractor dan ablation dengan input/budget sebanding selesai. Jangan
mengklaim graph reasoning dari keberadaan embedding tipe node saja.

## Acceptance sebelum training riil

- Rekonstruksi snapshot buggy sesuai adapter sumber; SWE-smith membutuhkan explicit
  instance reference dan pembalikan bug patch. Verifikasi target repair berlaku.
- Train/dev/eval split dan overlap audit dapat diulang dari manifest.
- Tokenizer/Jinja source dan special-token behavior sesuai checkpoint.
- Gold patch dan artifacts evaluator tidak muncul pada encoder atau prefix.
- Prefix/pad loss nol, target shifted sekali, lengths/position IDs konsisten.
- Collator synthetic lulus bukan bukti D03/D04 siap untuk data riil.

## Standalone SFT notebooks dengan cell seragam

[Training notebooks](TRAINING_NOTEBOOKS.md) menyediakan jalur pilot paired-file
editing → pretrained attention LoRA SFT → held-out CE → adapter/optimizer/RNG/cursor
checkpoint resume. Keluarga simetris/asimetris memakai 17 cell yang sama, dengan
16 isi cell identik. Shared data/training helpers telah diuji pada tiny fixtures;
RUN_TRAINING default False. CLI production train_direct_sft tetap placeholder.
