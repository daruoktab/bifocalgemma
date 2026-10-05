# Gemma 4 E2B/E4B: Audit Seluruh JSON Base dan Pembanding IT

Diperiksa **5 Oktober 2026**. Kedua repositori base yang diminta masing-masing memiliki
lima JSON: `config.json`, `generation_config.json`, `processor_config.json`,
`tokenizer.json`, dan `tokenizer_config.json`. Seluruh **10 JSON base** dibaca penuh,
termasuk tokenizer sekitar 32 MB; kelima JSON dari masing-masing varian IT juga diperiksa.
Tidak ada file safetensors diunduh atau bobot pretrained dimuat.

[Manifest audit](../tests/fixtures/gemma4_json_audit_manifest.json) merekam revision,
ukuran byte, SHA-256, key tingkat atas, statistik tokenizer, perbedaan base/IT,
penghitungan parameter di meta, dan pemeriksaan tokenizer runtime.
Snapshot config/metadata base: [E2B](../tests/fixtures/gemma4_e2b_base_config.json)
dan [E4B](../tests/fixtures/gemma4_e4b_base_config.json). Tokenizer besar disimpan
di cache audit lokal; tidak dimasukkan ke Git. Snapshot tidak menyalin bobot model.

## Hasil per file

| JSON | Kesimpulan audit | Implikasi untuk BifocalGemma |
| --- | --- | --- |
| `config.json` | Text config base identik dengan IT untuk ukuran yang sama; perbedaan root adalah EOS IT. Tower audio/vision tetap ada. | Encoder base dan decoder IT memakai struktur serta vocabulary yang kompatibel; hal ini tidak membuktikan bobot atau kualitasnya identik. |
| `generation_config.json` | Base berhenti pada ID 1; IT pada [1, 106, 50]. Sampling default sama: temperature 1.0, top-k 64, top-p 0.95, do_sample true. | Jangan membuat konfigurasi generasi IT hanya dari EOS pada model config. Pin stopping policy dan decoding budget tiap eksperimen. |
| `processor_config.json` | Sama secara byte di seluruh empat repo; memuat preprocessing gambar, audio dan video. | Untuk tahap text-only, tokenizer cukup; jangan memasukkan biaya tower multimodal sebagai biaya text backbone. |
| `tokenizer.json` | Base E2B/E4B sama; IT E2B/E4B sama. Model vocabulary/merges sama antar base dan IT, tetapi post-processor BOS berbeda. | Token IDs kode dapat sama, sementara tokenisasi dengan special tokens berbeda; hindari BOS ganda. |
| `tokenizer_config.json` | Base memiliki special-token mapping/padding; IT menambahkan response_schema dan response_template. E2B base juga menyebut add_bos_token true secara eksplisit. | Gunakan metadata decoder IT untuk parsing thinking/tool calls; metadata base bukan chat protocol yang sama. |

Sumber lengkap:
[base E2B](https://huggingface.co/google/gemma-4-E2B/tree/main),
[base E4B](https://huggingface.co/google/gemma-4-E4B/tree/main),
[IT E2B](https://huggingface.co/google/gemma-4-E2B-it/tree/main),
[IT E4B](https://huggingface.co/google/gemma-4-E4B-it/tree/main).
Revision yang digunakan tercatat pada manifest; link `main` bisa berubah.

## Struktur text backbone dan parameter yang tersembunyi oleh ringkasan config

| Besaran | E2B | E4B |
| --- | --- | --- |
| Hidden dimension | 1536 | 2560 |
| Layer lokal/global | 28 / 7 | 35 / 7 |
| Query heads / KV heads | 8 / 1 (MQA) | 8 / 2 (GQA) |
| Head dimension lokal/global | 256 / 512 | 256 / 512 |
| KV-shared layers (indeks nol) | 20, mulai layer 15 | 18, mulai layer 24 |
| MLP intermediate yang benar-benar dibangun | 6144 pada layer 0–14; 12288 pada 15–34 | 10240 pada seluruh 42 layer |
| PLE packed embedding shape | [262144, 8960] | [262144, 10752] |
| Text parameters unik, dihitung di meta | 4.628.569.344 | 7.463.013.376 |
| Vision + projection parameters | 168.544.256 | 169.330.688 |
| Audio + projection parameters | 307.183.904 | 308.756.768 |
| Seluruh multimodal parameters unik | 5.104.297.504 | 7.941.100.832 |

Hitungan berasal dari `Gemma4ForConditionalGeneration` Transformers **5.18.0**
yang diinstansiasi pada `meta`, dengan config base lengkap dan tied parameters
dihitung sekali. Tidak ada storage bobot dialokasikan. Hitungan ini bersifat
implementation-derived; buffer bukan parameter, dan precision checkpoint tidak
ditentukan hanya dari bentuk tensor.

`use_double_wide_mlp=true` pada E2B memperbesar MLP hanya pada KV-shared layers,
menurut [implementasi Gemma4TextMLP versi 5.18.0](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/gemma4/modeling_gemma4.py). Mengalikan 6144 dengan semua layer memberi
anggaran parameter yang salah. Tes meta kini memeriksa lebar MLP aktual ini.

`attention_k_eq_v=false` membuat K/V terpisah pada kedua Edge model. RoPE lokal
memakai theta 10000; global memakai proportional RoPE theta 1000000 dan partial
factor 0.25. Ini tidak otomatis berarti KV berkurang 37,5%. Global head dimension
justru 512, berbeda dari local 256; estimasi cache harus memodelkan kedua jenis
layer, window, sharing, backend, dan state transient.

Decoder language model tetap kausal secara default. Nama kelas
`Gemma4ForConditionalGeneration` tidak menjadikannya encoder-decoder untuk teks.
Adaptasi encoder bidirectional dan training yang sesuai masih diperlukan.

## Dampak pada anggaran bobot

Jika seluruh parameter text disimpan BF16, tanpa quantization, offload, atau sharing
antar backbone, biaya bobot matematis adalah:

| Pasangan encoder/decoder | Jumlah parameter text | Bobot dua byte/parameter, GB desimal |
| --- | --- | --- |
| E2B + E2B | 9.257.138.688 | 18,514 GB |
| E2B + E4B / E4B + E2B | 12.091.582.720 | 24,183 GB |
| E4B + E4B | 14.926.026.752 | 29,852 GB |

Ini belum mencakup bridge, optimizer, gradient, encoder states, KV, activation,
buffer, atau overhead allocator. Angka parameter efektif 2,3B/4,5B tidak dapat
menggantikan hitungan storage ini. Menghapus audio/vision membantu tetapi tetap
menyisakan text backbone fisik yang besar.

**Inferensi desain:** current bridge memakai multi-head K/V penuh, sedangkan
backbone memakai MQA/GQA dan KV sharing. Jika bridge dipasang di banyak layer
dengan cache terpisah, biaya cross-KV dapat besar. Penghematan total VRAM belum
terbukti; grouped/shared cross-KV, latent compression atau offloading merupakan
opsi eksperimen yang harus dinilai terhadap kualitas, bukan manfaat otomatis.

## Tokenizer, BOS, EOS, dan protocol

Tokenizer JSON menyatakan **BPE**, dengan 262144 vocabulary entries, 514906 merges,
dan 24 added tokens yang ditandai special. Normalizer mengganti spasi dengan
karakter U+2581; decoder memakai Replace, ByteFallback dan Fuse. Ini informasi
pipeline tokenizer, bukan bukti model sudah dilatih untuk code repair.

Vocabulary dan merges sama pada seluruh empat varian. Pemeriksaan runtime memakai
GemmaTokenizer dari aset lokal: tiga sampel kode dengan indentation, tab, whitespace,
dan Unicode round-trip tanpa perubahan pada masing-masing varian. Tanpa special
tokens, semua varian memberi token IDs yang sama pada sampel tersebut.

Dengan `add_special_tokens=True`, base menambahkan BOS ID 2, sementara IT tidak.
BOS untuk percakapan IT datang dari chat template. E4B base tidak menyebut
`add_bos_token` secara eksplisit, tetapi post-processor tokenizer.json dan hasil
runtime tetap menambahkan BOS. Jangan menyimpulkan perilaku dari satu flag saja.

| Token | ID | Peran metadata |
| --- | --- | --- |
| `<pad>` / `<eos>` / `<bos>` | 0 / 1 / 2 | Padding, akhir sequence, awal sequence |
| `<|tool_call>` / `<tool_call|>` | 48 / 49 | Pembuka/penutup tool call |
| `<|tool_response>` / `<tool_response|>` | 50 / 51 | Pembuka/penutup tool response |
| `<|channel>` / `<channel|>` | 100 / 101 | Pembatas channel, termasuk thinking |
| `<|turn>` / `<turn|>` | 105 / 106 | Pembuka/penutup turn |

Model config IT memiliki EOS [1,106], tetapi generation_config IT mencakup token
50 tambahan untuk boundary tool-response. Loader harus menghormati generation
config yang tepat. Tool executor/parsing harus membedakan patch, thinking, dan
tool calls; response_schema IT menyebut parser Gemma-specific dan escaped string
delimiters, sehingga output tool-call tidak diasumsikan selalu JSON standar.

`model_max_length` tokenizer bernilai sekitar 1e30 dan bukan context window
backbone. Batas text config adalah 131072. Semua tokenizer config memakai
`padding_side=left`; mask attention dan loss mask tetap perlu disiapkan.

Tidak ada chat_template dalam kelima JSON base maupun IT. Varian IT menyediakan
file pendamping **chat_template.jinja**, sedangkan repo base tidak. File Jinja
pendamping juga diperiksa sebagai pelengkap audit JSON. Loader tokenizer lokal
memiliki chat template hanya setelah aset IT Jinja tersedia. Training trajectory
tidak boleh sekadar memakai format prompt base untuk decoder IT.

## Aset multimodal

Vision config: hidden size 768, 16 layer, 12 heads, patch size 16, pooling kernel 3.
Processor gambar default memakai 280 soft tokens, sedangkan video memakai 32 frame
dan 70 soft tokens sebagai default preprocessing. Nilai default tidak diperlakukan
sebagai jaminan semua resolusi atau panjang video didukung oleh eksperimen ini.

Audio config: hidden size 1024, 12 layer dan 8 heads, dengan output_proj_dims
1536 pada kedua model. Projection embed_audio berikutnya memetakan output tower
1536 ke hidden text 1536 (E2B) atau 2560 (E4B). Feature extractor memakai 16 kHz, 128 features,
frame length 320, hop length 160, FFT 512; processor mencatat 750 audio tokens
dan 40 ms per token. Seluruh config memakai dtype bfloat16 dan tidak menyatakan
quantization_config; precision storage yang terukur tetap memerlukan audit checkpoint.

## Keputusan untuk eksperimen awal

- Gunakan base E2B/E4B sebagai encoder dan E2B/E4B IT sebagai decoder untuk
  seluruh empat pasangan dalam [desain eksperimen](EXPERIMENT_DESIGN.md),
  dengan revision/config/tokenizer masing-masing dipin. Pilot dimulai pada E2B–E2B IT. Token space kompatibel
  tetapi formatting BOS dan stopping policy harus mengikuti peran model.
- Pertahankan fokus text-only code repair. Tower multimodal dan agent tool protocol
  perlu implementasi tersendiri, bukan diasumsikan sudah tersedia.
- Laporkan physical/trainable parameters dan peak VRAM terukur; jangan memakai
  parameter efektif sebagai klaim iso-memory atau penghematan total.
- Pengujian config/shape pada meta, versi kecil full-depth, serta tokenizer runtime
  tidak membuktikan kualitas patch atau kemampuan pretrained pada task tertentu.

## Verifikasi audit

Seluruh 24 unit test lulus tanpa skip setelah penambahan pemeriksaan kesetaraan
text config base/IT, stopping policy, dan lebar MLP per layer pada meta. Ruff,
format, type checking, serta tautan lokal juga diperiksa. Tokenizer runtime
memverifikasi 12 round-trip sampel total (3 pada masing-masing 4 varian).

## Pemakaian audit pada scaffold aktif

[Registry C01](../src/bifocalgemma/config.py) menggunakan revisions audit ini.
Factory M04/M05 masih placeholder; direct merged architecture dan SFT sekarang
menjadi target. Word tying simetris dilakukan pada initialization dengan anchor
IT, PLE separate, tanpa teacher wajib. Angka bobot untied di audit adalah baseline
aritmetis yang tetap berlaku untuk asumsi tanpa sharing; model akhir perlu diukur
melalui [profiling](TRAINING.md). Latest checks ada di [VALIDATION](VALIDATION.md).
