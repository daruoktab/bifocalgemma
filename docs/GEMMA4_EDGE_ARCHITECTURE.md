# BifocalGemma: Referensi Backbone Gemma 4 Edge

Sumber berikut diperiksa pada **5 Oktober 2026**. Link `main` dapat berubah;
saat eksperimen, pin revision checkpoint dan simpan config yang benar-benar digunakan.

- [Config E2B IT](https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json)
- [Config E4B IT](https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json)
- [Model card Gemma 4](https://ai.google.dev/gemma/docs/core/model_card_4)

## Nilai config text backbone

| Field | E2B IT | E4B IT |
| --- | --- | --- |
| `architectures` (top-level) | `Gemma4ForConditionalGeneration` | `Gemma4ForConditionalGeneration` |
| `model_type` (text) | `gemma4_text` | `gemma4_text` |
| `hidden_size` | 1536 | 2560 |
| `intermediate_size` | 6144 | 10240 |
| `num_hidden_layers` | 35 | 42 |
| `num_attention_heads` | 8 | 8 |
| `num_key_value_heads` | 1 | 2 |
| `head_dim` / `global_head_dim` | 256 / 512 | 256 / 512 |
| `sliding_window` | 512 | 512 |
| Local/global layers dari `layer_types` | 28 / 7 | 35 / 7 |
| `num_kv_shared_layers` | 20 | 18 |
| `attention_k_eq_v` | `false` | `false` |
| `hidden_size_per_layer_input` | 256 | 256 |
| `use_double_wide_mlp` | `true` | `false` |
| `max_position_embeddings` | 131072 | 131072 |
| `vocab_size` | 262144 | 262144 |
| `use_bidirectional_attention` | `null` (default causal) | `null` (default causal) |

`Gemma4TextConfig` natively supports `use_bidirectional_attention="all"`. Pada checkpoint default HF, nilainya `null` (kausal). Untuk encoder repositori BifocalGemma, parameter ini disetel ke `"all"` sehingga seluruh token kode saling memperhatikan secara dua arah tanpa membutuhkan modifikasi kernel kustom.

`text_config.rope_parameters` memakai default RoPE untuk sliding layers dan
proportional RoPE untuk global layers dengan `partial_rotary_factor=0.25`.
Nilai konfigurasi itu sendiri tidak membuktikan persentase penghematan KV-cache.
Perilaku attention dan KV sharing harus diperiksa pada versi backend yang dipakai.

Config checkpoint mencatat `transformers_version=5.5.0.dev0`. Modul BifocalGemma
telah memverifikasi integritas struktural dimensi hidden E2B (1536) dan E4B (2560)
secara sintetis dengan `Gemma4TextModel` dan `GatedCrossAttentionBridge`.

## Parameter, checkpoint, dan memori

Model card membedakan **E2B: 2.3B efektif / 5.1B termasuk embeddings** dan
**E4B: 4.5B efektif / 8B termasuk embeddings**. PLE berisi tabel besar untuk
lookup token; parameter efektif tidak dapat langsung dikalikan dua untuk menyatakan
seluruh ukuran checkpoint atau resident VRAM.

| Besaran | Cara pelaporan yang diperlukan |
| --- | --- |
| Parameter efektif | Angka model card dengan definisinya |
| Parameter fisik | Jumlah tensor backbone yang benar-benar dimuat, termasuk bridge |
| Parameter trainable | Parameter yang menerima update pada konfigurasi adaptasi |
| Ukuran checkpoint | Ukuran file sebenarnya, dtype/quantization, dan komponen yang disimpan |
| Resident weights | Bobot pada GPU/CPU, dengan sharing dan offloading yang eksplisit |
| Peak VRAM | Pengukuran allocator untuk panjang konteks, batch, training/inference, dan hardware tertentu |

Sebagai aritmetika kasar, 5.1B elemen pada dua byte memerlukan sekitar 10.2 GB
desimal dan 8B sekitar 16 GB, sebelum overhead. Itu bukan ukuran checkpoint text-only
yang terukur, bukan total VRAM training, dan bukan biaya pasangan encoder-decoder.
Menghapus modul multimodal, berbagi bobot, atau offloading mengubah biaya aktual.

Tidak ada jaminan model ganda muat di GPU 24 GB atau dua T4. Memori dua GPU tidak
otomatis menjadi satu ruang alamat; metode pembagian bobot dan aktivasi perlu dirancang.

## Implikasi integrasi BifocalGemma

- Encoder yang diadaptasi dapat mengaktifkan bidirectional attention secara native via `Gemma4TextConfig(use_bidirectional_attention="all")`, yang telah diverifikasi mengalirkan konteks global ke token awal dan menghubungkan gradien ke bridge.
- [Bridge](../src/bifocalgemma/models/bridge.py) dapat menerima dimensi berbeda,
  tetapi kompatibilitas residual, normalization, posisi layer, dan cache masih harus diuji.
- [AST projector](../src/bifocalgemma/models/code_graph_projector.py) menerima ID
  tipe/kedalaman. Dimensi integrasi adalah 1536 untuk E2B dan 2560 untuk E4B;
  default generiknya bukan konfigurasi backbone yang otomatis terhubung.
- **BifocalGemma-E2B-E2B** menjadi kandidat awal; konfigurasi asimetris dan E4B-E4B
  ditunda. Parameter efektif yang berdekatan tidak berarti budget memori identik.

[T5Gemma](https://developers.googleblog.com/en/t5gemma/) menjadi referensi adaptasi
encoder-decoder, termasuk konfigurasi asimetris. Klaim kontribusi baru harus mengacu
pada metode atau bukti perbaikan kode, bukan sekadar pemisahan encoder dan decoder.
