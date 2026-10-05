# BifocalGemma: Ikhtisar Spesifikasi dan Status

**Paper:** *BifocalGemma: Evaluating Bidirectional Repository Encoding for Local Code Repair*

Rancangan aktif adalah encoder base + decoder IT, **merged attention sejak awal**
dan direct SFT. Word tying hanya pada simetris; PLE terpisah. Distillation,
post-training conversion dan teacher network adalah opsi, bukan pipeline wajib.

## Dokumen sumber per subsistem

| Bagian | Spec utama | Kode |
| --- | --- | --- |
| Enam model dan RQ | [EXPERIMENT_DESIGN](EXPERIMENT_DESIGN.md) | [C01](../src/bifocalgemma/config.py) |
| Tensor/attention/init | [ARCHITECTURE](ARCHITECTURE.md) | [M01](../src/bifocalgemma/models/merged_attention.py), [M04](../src/bifocalgemma/models/initialization.py) |
| Data dan masking | [DATA_PIPELINE](DATA_PIPELINE.md) | [D01](../src/bifocalgemma/contracts.py), [D02](../src/bifocalgemma/data/collator.py) |
| Objective/trainer | [TRAINING](TRAINING.md) | [T01](../src/bifocalgemma/training/objectives.py), [T02](../src/bifocalgemma/training/trainer.py) |
| Inference/evaluation | [EVALUATION](EVALUATION.md) | [I01](../src/bifocalgemma/inference/generation.py), [E01](../src/bifocalgemma/evaluation/harness.py) |
| Readiness/urutan | [BUILD_GUIDE](BUILD_GUIDE.md) | [Registry blok](../src/bifocalgemma/scaffold.py) |

## Kontrak minimum

Seq2SeqBatch menyediakan encoder/decoder IDs dan validity masks, labels -100 pada
prefix/pad, serta position IDs yang kompatibel dengan left padding. Encoder
mendapat context sebelum fix dan issue. Decoder mendapat prefix IT + target
shifted hanya pada training. M05 mengembalikan ModelOutput(logits,loss).

M01 reference memakai satu softmax pada concatenated self/encoder keys, shared
K/V projections, GQA, serta H adapter untuk asimetris. Mask tetap causal untuk
decoder dan globally visible untuk encoder keys. Target production mempertahankan
native RoPE, PLE, MLP, normalization dan KV-sharing; komponen itu belum terintegrasi.

M02 sharing menggunakan Parameter alias, dengan decoder IT sebagai anchor
ordinary words. Optimizer/checkpoint harus menjaga alias dan PLE tetap separate.
M03 tiny assembly hanya reference; native test snapshots terpisah dari pipeline.

## Boundary yang disengaja

Placeholder mengeluarkan FeatureNotReadyError dengan ID dan tautan docs.
Factory/trainer/generator/harness tidak menyembunyikan fallback synthetic atau
mencetak score nol seolah pengukuran. Complete signatures dan acceptance criteria
ada dalam BUILD_GUIDE. Printed table/prediction serializer tidak menjalankan benchmark.

Tidak ada bobot pretrained/dataset besar diunduh untuk local validation. Hasil
checks aktual dicatat pada [VALIDATION](VALIDATION.md); cek modul bukan bukti
convergence atau kualitas repair.

## Batas estimasi KV lama

[RepoContextRouter](../src/bifocalgemma/models/routing.py) tetap utilitas reference.
calculate_decoder_self_kv_savings memakai full-attention layers seragam:

```text
C = 2 * layers * KV_heads * head_dim * bytes_per_element
baseline_self = (prompt + generation) * C
split_self = generation * C
```

Key *_mb berarti MiB. Defaults adalah ilustrasi, bukan Gemma config. Fungsi tidak
memodelkan weights, Henc, cross-KV, activations, local/global head dimensions,
windowing, KV-sharing atau allocator peaks. create_bidirectional_repo_mask hanya
mask token valid, bukan pengubah causal/window kernel. Tidak ada klaim penghematan
total >95% atau keunggulan pasti dibanding decoder-only.

## Memory dan data

Ordinary tying memberi bobot matematis text simetris sekitar 17,709/28,510 GB
BF16 sebelum biaya model akhir/activation/optimizer; asimetris sekitar 24,183 GB
sebelum adapter. Exact native counts dan source fingerprints ada pada
[audit JSON](GEMMA4_JSON_AUDIT.md). Parameter efektif 2,3B/4,5B bukan total storage.

Data source/filtering mengikuti [riset dataset](DATASETS_AND_BENCHMARKS.md).
Graph-UL2/contrastive localization/AST features/multimodal/partial invalidation
bukan kemampuan yang otomatis hadir pada v0. Mereka harus memiliki blok dan
ablation tersendiri bila diaktifkan.
