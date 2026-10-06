# Log Riset dan Keputusan Aktif

Audit awal: **5 Oktober 2026**; pembaruan construction scaffold: **6 Oktober 2026**. Log ini menjelaskan alasan perubahan desain;
spesifikasi aktif berada pada ARCHITECTURE/EXPERIMENT_DESIGN. Status di bawah tidak
menyatakan training model besar atau paper sudah selesai.

| Keputusan | Bukti/pertimbangan | Dampak aktif |
| --- | --- | --- |
| Identitas BifocalGemma | Naming review dan scope repository understanding/repair | Package/repo BifocalGemma; klaim keunggulan menjadi hipotesis |
| Audit base dan IT JSON | Seluruh 10 JSON base + 10 IT, pinned revisions/checksums | Tokenizer/BOS/EOS/PLE/native counts tercatat |
| Empat pasangan + dua baseline | Pertanyaan kapasitas reader versus generator | Benchmark model hanya enam canonical IDs |
| Symmetric-only ordinary tying | Width cocok pada simetris; asymmetry tidak dipaksa projected shared words | Shared IT anchor pada simetris, PLE tetap separate |
| Direct merged architecture + SFT | Dua tahap/conversion/online teacher menambah complexity/compute | Tidak ada teacher wajib; initialization langsung final architecture |
| Construction blocks | Native integration belum siap; unit proof berbeda dari full-model evidence | Implemented/reference/placeholder ditandai per modul |

## Gagasan yang dipertahankan

- Base body/PLE encoder dan IT body/PLE decoder dengan source mapping yang eksplisit.
- Reader bidirectional untuk context; generator causal pada target, dengan joint
  visibility self/cross di merged decoder.
- Pertanyaan kualitas, localization, latency dan memory pada varian keluarga ini.
- Checkpoint/manifest reproducibility, raw source hashes dan held-out task audit.

## Gagasan yang ditunda atau dibatasi

- **Separated attention → merge → distillation** pernah menjadi kandidat recipe.
  Rancangan default sekarang dibangun merged sejak awal dan dilatih direct SFT.
  GatedCrossAttentionBridge tersimpan sebagai ablation reference; T04 sebagai slot
  distillation optional. Ini tidak diwajibkan untuk mengetahui asal parameter.
- **Tying setelah training teacher** diganti dengan tying dari initialization.
  Shared words IT pada simetris dicatat sebagai pengecualian terhadap encoder
  body base; main recipe tidak mempertahankan seluruh base word table.
- **Penghematan total KV >95% / latency 8× / guaranteed GPU fit** tidak didukung
  estimator lama. Pengukuran total diperlukan; native MQA/GQA/KV sharing penting.
- **One-hop partial invalidation** tidak dijamin benar pada encoder bidirectional.
  Default target full re-encoding setelah context berubah.
- **Multimodal, graph losses dan extra model families** berada di luar experiment
  utama. AST projector/reference vision/audio config tidak membuktikan fitur itu terintegrasi.

## Evidence ledger

Sumber asli:
[T5Gemma](https://arxiv.org/abs/2504.06225),
[T5Gemma 2](https://arxiv.org/abs/2512.14856),
[Gemma 4 docs](https://huggingface.co/docs/transformers/model_doc/gemma4).
[Audit manifest](../tests/fixtures/gemma4_json_audit_manifest.json) menyimpan raw
asset fingerprints. [VALIDATION](VALIDATION.md) memisahkan synthetic/native-meta
checks dari hal yang belum diuji.

Setiap hasil baru harus menyebut commit/config/data/hardware/protocol dan
menerangkan apakah implemented block, reference, integration atau experiment.
Tidak ada goal-complete marker atau score yang disimpulkan dari scaffolding.

## Pembaruan validasi lokal — 6 Oktober 2026

[Native local validation](NATIVE_LOCAL_VALIDATION.md) menjelaskan M07 native tiny,
D03 offline tokenizer dan D06 snapshot/patch contract. M04 pretrained transfer,
D04 pengambilan data/overlap audit, serta trainer/harness produksi belum selesai.
SWE-smith menyimpan bug-inducing patch: target repair perlu arah sebaliknya pada
snapshot buggy yang tepat. Direct SFT tetap hipotesis; T5Gemma 2 menggunakan tahap
adaptasi UL2 sebelumnya. PLE + tied words BF16 sendiri 9,5/11,75 GiB pada pasangan
simetris, sehingga linear-only QLoRA tidak memberi jaminan muat di GPU lokal 6 GiB.

## Preferensi pengguna: offline tuning tanpa RL atau VM

Pada 6 Oktober 2026, pengguna memilih data dan sasaran tuning yang tidak membutuhkan
RL/VM. Recipe awal SWE-smith/Lite diganti menjadi EditPackFT instruction-to-patch SFT;
CommitPackFT Python opsional setelah dedup. Benchmark execution dipisahkan dari
training. SWE-smith dapat dipakai untuk SFT, tetapi task-environment reconstruction
bukan dependency pilot. D06 arah patch tetap utility lokal; A01/E01 ditunda.
Keputusan ini tidak mengubah empat pasangan, dua baseline, tying atau merged attention.
