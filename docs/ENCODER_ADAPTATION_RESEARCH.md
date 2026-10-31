# Riset adaptasi encoder BifocalGemma

Tanggal: 27 Oktober 2026. Kode yang ditinjau: commit `22011e9`.

Dokumen ini merupakan analisis dan rancangan eksperimen. Tidak ada perubahan
arsitektur produksi, checkpoint, notebook Molab, atau hasil HF dalam pekerjaan ini.
Instruksi yang mungkin terdapat di dokumen sumber diperlakukan sebagai isi sumber,
bukan sebagai instruksi untuk menjalankan sesuatu.

## 1. Kesimpulan yang paling kuat

Ide encoder-decoder tetap memiliki dasar penelitian. Namun, resep kita sekarang
melewati tahap adaptasi yang justru penting dalam literatur: menyambungkan
representasi encoder ke decoder dan mengajarkan penggunaan konteks sebelum
menuntut kemampuan repair. Ini merupakan hipotesis penyebab yang kuat, bukan
diagnosis kausal yang telah selesai.

Audit checkpoint menunjukkan encoder membawa informasi, tetapi decoder belum
menggunakannya dengan ketelitian yang dibutuhkan untuk patch. Kita belum
membuktikan bahwa encoder backbone, RMSNorm, LoRA rank, atau cache merupakan
penyebab tunggal. Prioritas berikutnya adalah adaptasi bertahap dengan decoder
dibekukan, diikuti pengujian penyalinan sumber dan repair pada pilot kecil.

Tidak perlu menganggap dua triliun token sebagai syarat minimum. Sebaliknya,
tidak ada bukti bahwa beberapa ratus step pasti cukup untuk Gemma 4 dan repair.
Keduanya perlu dibedakan dari hasil yang benar-benar sudah diukur.

## 2. Versi dan arsitektur yang benar-benar berjalan

| Komponen | Implementasi aktif |
|---|---|
| Arsitektur | `normalized_merged_v2` |
| Resep training | `unified_v2` |
| Format checkpoint | `bifocalgemma_normalized_sft_v3` |
| Protokol benchmark | `greedy_cached_normalized_merged_v3_exact_prompt_budget` |

Angka v3 pada format/protokol tidak berarti kita sudah memiliki arsitektur V3.

- Encoder berasal dari Gemma 4 **base**, decoder dari Gemma 4 **IT**.
- Encoder menggunakan perhatian bidirectional. Pola local/global native tetap
  ada: local menjadi dua arah dalam jendela; bukan semua layer menjadi global.
- Word embedding encoder dan decoder terpisah, juga untuk pasangan simetris.
  Embedding decoder tetap tied dengan LM head. Word embedding dan PLE dibekukan.
- Encoder final hidden state diproyeksikan oleh linear tanpa bias dan RMSNorm
  FP32. Linear simetris dimulai identity, asimetris semi-orthogonal; gain norm satu.
- Decoder menggunakan **merged attention**, satu softmax gabungan self/cross dan
  proyeksi K/V pretrained yang sama. Tidak ada gate atau cross-K/V independen.
- Native Q/K/V normalization, scaling, residual, RoPE self, PLE, local mask, dan
  ownership KV tetap dipertahankan. Cross-key tidak diberi posisi RoPE decoder.
- Encoder/decoder attention LoRA rank 16 alpha 32 dan alignment dilatih bersama
  sejak awal. Tidak ada tahap khusus adaptasi representasi atau frozen decoder.
- Objective berupa SFT target tugas. Tidak ada tahap UL2/PrefixLM adaptation
  sebelum SFT. Thinking dimatikan pada jalur train/eval kita.

Sumber kode: [konfigurasi](../src/bifocalgemma/config.py),
[merged attention](../src/bifocalgemma/models/memory_efficient.py),
[alignment](../src/bifocalgemma/models/alignment.py),
[policy](../src/bifocalgemma/training/policy.py),
[recipe](../src/bifocalgemma/training/recipes.py).

Beberapa dokumen desain lama menjelaskan eksperimen legacy. Pernyataan tentang
arsitektur aktif di sini mengikuti kode dan recipe v2, bukan bagian legacy itu.

## 3. Apa yang telah dan belum dibuktikan audit

Pada tiga contoh HumanEvalFix yang dipilih untuk diagnosis, rata-rata NLL gold:

| Model | Konteks benar | Encoder kosong | Konteks tertukar |
|---|---:|---:|---:|
| E2B-E2B | 0.4815 | 0.6451 | 0.7647 |
| E4B-E4B | 0.3930 | 0.6155 | 0.8084 |

Konteks benar membantu keenam kasus paired. Karena itu, klaim "encoder sama
sekali tidak digunakan" bertentangan dengan pengukuran ini. Namun, output bebas
masih bisa menciptakan nama fungsi atau baris sumber yang salah. Loss
teacher-forced yang rendah tidak membuktikan fidelity saat autoregressive decoding.

Menonaktifkan cache dan memakai prompt training tidak menyelamatkan prediksi
paired yang diperiksa. Memasukkan sumber langsung ke decoder membantu beberapa
kasus; hasil itu mengubah jalur input dan hanya merupakan kontrol diagnostik.
Kontrak prompt unified diff versus parser git-diff yang lebih ketat juga perlu
diselaraskan. Perbaikan hitungan hunk saja tidak memperbaiki sumber yang dikarang.

Audit ini bukan sampel statistik representatif dan belum mencakup pasangan
asimetris. Lihat [bukti lengkap dan batasannya](CHECKPOINT_AUDIT.md).

## 4. Empat paper: bagian yang dapat dipindahkan ke proyek kita

### Encoder-Decoder Gemma / T5Gemma

Paper ini mengadaptasi Gemma 2 menggunakan PrefixLM atau UL2 sebelum instruction
post-training. Untuk pasangan tidak seimbang, cross-attention dilatih lebih dahulu
sementara parameter lain dibekukan. Warmup membantu, tetapi memperpanjangnya dari
1K ke 5K step justru menurunkan hasil pada ablasinya. Pelajarannya adalah adaptasi
bertahap perlu diuji; angka warmup paper bukan optimum untuk LoRA kita.
Skala adaptasinya mencapai 2T token. [Paper, bagian 3 dan 6](https://arxiv.org/html/2504.06225v1).

### T5Gemma 2

Ini dasar yang relevan untuk merged attention: softmax self/cross bersama dan
parameter attention bersama memang merupakan desain penelitian yang sah.
Namun, model ini berbasis Gemma 3 dan diadaptasi dengan sekitar 2T token UL2
sebelum post-training. Ablasi merged attention adalah penghematan parameter
dengan sedikit penurunan kualitas, bukan bukti bahwa merged selalu lebih akurat.
Tying embedding mereka juga bukan alasan otomatis untuk mengganti embedding
base encoder kita dengan IT. [Paper, bagian 2-3](https://arxiv.org/html/2512.14856v2).

### Gemma 4 Technical Report

Paper ini mendefinisikan backbone dan kontrak native yang harus dipertahankan.
Model IT telah menjalani post-training dan memiliki kontrol thinking, tetapi
itu bukan pelatihan untuk menjadi encoder bidirectional. Tidak adanya trace
thinking pada dataset kita sendiri belum membuktikan ketidakcocokan: mode
non-thinking tersedia. Ini tidak menjelaskan semua kegagalan sumber pada audit.
[Technical report](https://arxiv.org/abs/2607.02770),
[model card resmi](https://ai.google.dev/gemma/docs/core/model_card_4).

### AF-Muon

Kontribusinya terutama menyangkut optimisasi berdasarkan peran parameter,
termasuk tabel vocabulary tied yang menerima dua jenis gradien. Dalam resep
kita tabel tersebut dibekukan; trainable utamanya LoRA dan alignment. Mengganti
AdamW tidak mengajarkan pemetaan encoder-decoder yang belum terbentuk.
Prioritas paper ini untuk masalah sekarang rendah. Aturan untuk matriks dense
juga tidak boleh dipindahkan tanpa analisis ke dua faktor LoRA.
[Paper](https://arxiv.org/abs/2610.01395).

## 5. Penelitian tambahan yang paling relevan

| Penelitian | Temuan yang relevan | Batas transfer ke BifocalGemma |
|---|---|---|
| [CodeT5+](https://aclanthology.org/2023.emnlp-main.68/) | Model kode dengan decoder pretrained dibekukan, encoder kecil dan cross-attention trainable; objective mencakup denoising dan causal LM. | Dukungan langsung untuk freeze pada model kode; tetap memakai adaptasi data besar, bukan bukti pilot kecil pasti menang. |
| [LLM2Vec](https://arxiv.org/html/2404.05961) | Membuka bidirectional lalu adaptasi masked next-token prediction dengan LoRA. MNTP paper menggunakan 1.000 step; hasil bergantung backbone. | Embedding, bukan repair. Memberi dasar adaptasi murah, bukan estimasi durasi kita. |
| [LangBridge](https://arxiv.org/html/2401.10695) | Encoder dihubungkan dengan linear ke LM frozen; 200 ribu contoh pada eksperimen utama. Kebijakan freeze encoder bergantung jenis LM/data. | Encoder pretrained bidirectional dan jalur soft-prefix berbeda dari merged Gemma 4. Linear sederhana bisa bekerja, tetapi tidak untuk setiap checkpoint. |
| [Flamingo](https://arxiv.org/html/2204.14198) | Cross-attention baru memakai gate tanh awal nol di antara blok LM frozen. | Vision-language dan data sangat besar; prinsip menjaga fungsi awal dapat dipakai, angka performa tidak dapat dipindahkan. |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | Bridge dilatih bertahap di antara backbone frozen. | Kompresi ke sedikit query dapat menghilangkan detail identifier kode. Jangan menyalin bottleneck vision secara buta. |
| [Look Both Ways and No Sink](https://aclanthology.org/2025.acl-long.1113/) | Konversi encoder tanpa training bisa berhasil pada tugas embedding dengan perlakuan attention sink. | Menolak klaim bahwa bidirectional selalu memerlukan training panjang; bukan bukti sink adalah penyebab kegagalan repair kita. |
| [Warm-starting seq2seq](https://aclanthology.org/2020.tacl-1.18/) | Checkpoint encoder/decoder pretrained dapat digunakan untuk sequence generation. | Keberhasilan warm-start tidak membuat semua representasi lintas model langsung kompatibel. |

Kesamaan metodologinya: manfaat dari backbone pretrained tetap memerlukan
antarmuka dan objective yang sesuai. Tidak ada satu paper yang menjamin encoder
Gemma 4 base + decoder IT + satu epoch repair LoRA akan unggul.

## 6. Risiko arsitektur yang masih masuk akal

### 6.1 Pergantian mask mengubah representasi, meskipun parameter sama

Membuka akses ke token masa depan mengubah distribusi hidden state di setiap
layer. LoRA attention dapat menyesuaikannya, tetapi SFT target repair memberi
sinyal yang tidak langsung untuk membentuk representasi penyalinan token.
Ini hipotesis yang perlu dibandingkan dengan encoder causal pada budget sama.
Bidirectional tetap berpotensi berguna, khususnya infilling dan konteks setelah
lokasi edit; jangan membuangnya hanya karena satu konfigurasi gagal.

### 6.2 Norm dan shape tidak menyelesaikan kompatibilitas semantik

Linear identity lalu RMSNorm umumnya bukan pemetaan identity secara keseluruhan.
Semi-orthogonal menjaga sifat geometri tertentu, bukan menyamakan arti fitur.
Final hidden encoder juga berbeda dari input attention decoder di setiap depth.
Satu alignment yang dipakai semua layer membebani optimisasi, tetapi belum
terbukti terlalu sederhana. LangBridge justru memperingatkan agar tidak otomatis
menganggap MLP/resampler lebih baik.

Tidak ditemukan bukti baru bahwa RMSNorm saat ini salah secara implementasi.
Menghapus/menambah norm lagi tanpa pengukuran bukan rekomendasi utama.

### 6.3 Merged attention langsung mengubah fungsi decoder

Untuk satu query, definisikan Zs dan Ze sebagai jumlah eksponensial skor valid
pada self dan encoder. Os dan Oe adalah keluaran attention yang dinormalisasi
masing-masing. Maka, sebelum output projection:

`O_joint = (Zs * Os + Ze * Oe) / (Zs + Ze)`

Walaupun seluruh parameter decoder frozen, Ze tetap masuk penyebut. Encoder
value nol juga belum mengembalikan fungsi self-only, karena normalisasi berubah.
Encoder yang sepenuhnya dimask barulah menghilangkan kontribusi normalisasi ini.
Jumlah token sumber bisa memengaruhi kompetisi perhatian; dominance encoder
belum boleh diklaim tanpa mengukur mass attention per layer/head/query.

K/V LoRA bersama juga berarti update untuk membaca encoder mengubah proyeksi
self. Memisahkan hanya adaptor cross-K/V dapat diuji sebagai alternatif dengan
parameter tambahan terukur. Pada Gemma 4, adaptor harus mengikuti owner KV,
GQA, dan head dimension; shared layer tidak boleh membuat owner sendiri.

### 6.4 Mengikuti format patch berbeda dari membaca source

Patch-only loss dapat memberi reward untuk pola patch yang umum sementara
ketelitian token langka belum terbentuk. Teacher forcing memasok prefix gold,
sehingga loss yang tampak baik bisa menyembunyikan kesalahan yang menumpuk pada
generasi bebas. Perlu objective dan evaluasi yang memisahkan penyalinan,
lokalisasi, format, dan repair fungsional.

### 6.5 Posisi cross-key bukan kesalahan yang telah terbukti

Encoder hidden sudah mengandung informasi posisi. Cross-attention tidak harus
memakai RoPE decoder seperti self-attention. Tidak adanya cross RoPE saja tidak
membuktikan bug. Mengubahnya perlu eksperimen terkontrol karena indeks sumber
dan posisi target tidak mempunyai makna koordinat yang otomatis sama.

## 7. Apakah encoder IT lebih masuk akal?

Layak diuji, terutama pasangan simetris yang encoder/decoder dapat dimulai dari
checkpoint IT yang sama. Hipotesisnya adalah representasi awal lebih kompatibel
dan pengetahuan instruction/code sudah lebih dekat ke target. Itu belum menjadi
bukti; hidden state encoder akhir tetap tidak sama dengan hidden decoder pada
setiap layer, dan mask bidirectional tetap merupakan perubahan baru.

IT bukan sinonim encoder yang lebih baik. Post-training mengoptimalkan perilaku
generasi/instruction, bukan fidelity token bidirectional. Base bisa lebih baik
untuk adaptasi representasi tertentu. Hindari mengganti base ke IT bersamaan
dengan norm, gate, objective dan LoRA rank: hasilnya tidak mengisolasi penyebab.

Kontrol yang disarankan setelah tahap alignment dasar berjalan: base versus IT
encoder, dengan decoder, ukuran, tokenizer, format sumber, embedding ownership,
mask, adapter, contoh dan schedule yang sama. Pin revision keduanya. Jika encoder
IT dipakai, embedding tetap terpisah agar tidak sekaligus mengubah kebijakan tying.

IT juga tidak otomatis membutuhkan target chain-of-thought. Untuk eksperimen
non-thinking kita, kontrol token dan template train/eval harus konsisten.

## 8. Freeze segitiga bawah dan train segitiga atas

Matriks attention `[jumlah token, jumlah token]` adalah aktivasi yang dihitung
ulang untuk setiap input. Parameter Wq/Wk/Wv mempunyai sumbu fitur/head, bukan
sumbu posisi. Mask causal tidak memiliki bobot pretrained yang bisa dibekukan.

Misalnya, dua posisi lama memiliki skor sama dan probabilitas `[0.5, 0.5]`.
Ketika dua posisi atas dibuka dengan skor sama, probabilitas posisi lama menjadi
`[0.25, 0.25]`, walaupun skor lamanya tidak berubah. Mematikan gradien pada
sebagian attention edge tidak menghilangkan perubahan penyebut softmax atau
perubahan hidden state yang diteruskan ke layer berikutnya.

Analogi yang benar-benar dapat diimplementasikan adalah **dua cabang**:

`hidden_baru = hidden_causal_frozen + tanh(g) * branch_bidirectional_trainable`

Cabang tambahan dapat dibatasi hanya future edges atau memakai seluruh sumber.
Ini perubahan arsitektur dengan normalisasi/residual sendiri, bukan freeze
segitiga pada satu softmax. Saat g=0, cabang awal terjaga pada input identik;
setelah gate tumbuh, representasi layer berikutnya tetap dapat berubah.
Jangan membuka akses ke future gold token di decoder training: itu leakage.
Bidirectional hanya boleh atas source yang memang tersedia saat inference.

Pada Gemma 4, local layer memiliki upper band sesuai window, bukan upper
triangle penuh. Pendekatan ini mungkin diteliti nanti, tetapi lebih kompleks
dan bukan perubahan pertama yang paling mudah diatribusikan.

## 9. Freeze dan definisi apple-to-apple

Definisikan N sebagai jumlah update SFT decoder baseline dan A sebagai update
adaptasi encoder/bridge. Berikut klaim yang berbeda:

| Desain | Yang setara | Yang tidak setara |
|---|---|---|
| Baseline N; paired A frozen-decoder lalu N joint | Data/objective/budget SFT decoder dapat disamakan | Total compute dan paparan data paired lebih besar |
| Baseline N; paired A frozen-decoder lalu N-A joint | Total jumlah batch/paparan dapat disamakan | Decoder mendapat lebih sedikit update; FLOP per batch berbeda |
| Kedua model mendapat adaptasi A pada korpus terkait lalu SFT N | Mengontrol manfaat data/continual adaptation tambahan | FLOP dan parameter tetap berbeda |
| Keduanya dibatasi budget FLOP/GPU-hour | Anggaran resource dapat dibandingkan | Jumlah contoh, token dan decoder update dapat berbeda |

Jadi usulan user valid sebagai **perbandingan dengan budget SFT decoder sama**.
Tidak valid menyebutnya sekaligus setara total training hanya karena decoder
mendapat jumlah update yang sama. Bahkan pada N yang sama, gradien decoder
berbeda karena konteks melalui jalur yang berbeda.

Untuk penelitian yang kuat, laporkan minimal matched-decoder-SFT dan kontrol
baseline yang mendapat kesempatan memakai data adaptasi tambahan. Sertakan
GPU-hour/FLOP, token non-padding, encoder/decoder update, biaya teacher bila ada,
parameter total/trainable, VRAM dan biaya inference. E4B encoder + E2B decoder
bukan model dengan total parameter sama seperti E2B decoder-only.

Kontrol atribusi tambahan: gunakan **decoder baseline SFT yang sama persis dan
frozen** pada kedua pembacaan, lalu pelajari encoder/bridge. Ini menguji apakah
antarmuka dapat membawa informasi ke decoder yang diketahui bisa repair.
Tetap ada biaya adaptasi tambahan dan perbedaan jalur input. Jangan mengganti
baseline utama dengan kontrol tersebut atau mengklaim kesetaraan total.

## 10. Urutan eksperimen yang direkomendasikan

Tujuannya mengurangi risiko menghabiskan satu minggu pada semua pasangan sebelum
mengetahui apakah jalur informasi bekerja. Ini eksperimen pengembangan; hasil
akhir tetap dapat memakai satu resep dan enam notebook satu-Run.

### Langkah 0: kunci kontrak dan cohort diagnostik

- Samakan instruksi format patch dengan parser untuk semua keluarga model.
  Pisahkan format success, apply success, dan functional pass.
- Bekukan development cohort berisi copy span, identifier extraction, function
  localization, infilling sederhana, dan repair. Holdout berdasarkan proyek atau
  sumber sebelum membuat variasi contoh, agar variasi fungsi yang sama tidak bocor.
- Gunakan identifier acak, nama/path yang tidak lazim, dan perubahan nama sebagai
  kontrol agar keberhasilan bukan hanya hafalan. Jangan gunakan hidden tests atau
  output benchmark sebagai data training/pemilihan hyperparameter.
- Mulai dari E2B-E2B. Setelah metodologi terbukti, uji E4B-E4B untuk kapasitas dan
  memori, lalu pasangan asimetris. Stress test VRAM E4B-E4B tetap diperlukan.

### Langkah 1: alignment dengan decoder frozen, arsitektur merged tetap

Ini eksperimen pertama yang paling murah perubahan kodenya dan paling jelas
menguji hipotesis yang terlewat pada v2. Bekukan seluruh decoder, termasuk
LoRA-nya. Latih alignment, lalu encoder attention LoRA; gunakan awal bridge-only
singkat bila gradien encoder belum stabil. Tahapan harus tercatat pada manifest.

Objective awal yang disarankan merupakan campuran tugas yang mempunyai jawaban
terverifikasi langsung dari source: salin span/fungsi terpilih, keluarkan nama
fungsi/identifier secara tepat, lanjutkan prefix, serta pulihkan span source yang
disembunyikan. Masukkan beberapa panjang konteks dan posisi jawaban. Hindari
target yang menyalin seluruh file sebagai satu-satunya objective; itu dapat
memakan anggaran tanpa mengajarkan seleksi.

Denoising sungguhan harus menghilangkan target dari source. Sebaliknya, untuk
copy/extraction, target memang tersedia di source secara sengaja. Kedua jenis
tugas itu berbeda dan harus diberi label pada metrik. Gunakan delimiter/token
yang sudah tersedia dan tokenizer-consistent; jangan menambah sentinel acak yang
embedding-nya tetap frozen tanpa validasi.

Checkpoint decoder awal harus ditentukan di muka: IT resmi untuk calon resep
utama; baseline SFT frozen hanya kontrol atribusi terpisah. Jangan mencampurkan
keduanya dalam satu klaim training.

Pilot 200-500 optimizer step adalah usulan anggaran awal, bukan angka optimal
paper atau jaminan durasi. Untuk memilih recipe, bandingkan pilot yang dimulai
dari backbone, cohort, seed, dan total budget yang sama: direct joint versus
jadwal frozen-then-joint. Pilot tidak otomatis diteruskan menjadi full run.

Penting: decoder frozen tetap harus menyediakan autograd terhadap input encoder.
`no_grad()` di seluruh decoder memutus pelajaran encoder/bridge. Beku parameter
bisa menghemat state optimizer/gradiennya, tetapi tidak menghilangkan semua
aktivasi/backward; jangan menjanjikan penghematan waktu/VRAM tanpa pengukuran.

### Langkah 2: bila merged tetap mengganggu, baru isolasi jalur cross

Jika loss copy tidak turun atau mass encoder merusak pembacaan self, uji **satu
alternatif**: cross-attention residual dengan parameter cross terpisah dan gate,
sambil membekukan self-attention/FFN decoder. Ini mengikuti prinsip CodeT5+ dan
Flamingo, tetapi bukan lagi merged joint-softmax v2. Semua tambahan parameter,
latency, dan mekanisme KV perlu dilaporkan.

Contoh prinsipnya: `h <- h + tanh(g_l) * CrossAttn_l(h, Henc)`.
Gate nol menjaga keluaran awal pada input decoder yang identik, bukan menjamin
skor sama dengan baseline yang melihat raw source langsung. Jangan menginisialisasi
gate **dan** seluruh output branch nol sekaligus, karena dapat mematikan gradien
kedua sisi. Inisialisasi dan finite-gradient test harus diperiksa.

Versi yang mempertahankan joint softmax tetapi memberi bias logit cross atau
cross-specific LoRA juga mungkin, tetapi belum otomatis menjaga fungsi awal.
Gate sangat kecil dapat memperlambat pembelajaran encoder. Jangan menambah
semua mekanisme sekaligus.

### Langkah 3: objective tambahan hanya bila diagnosis menuntut

MNTP pada encoder, terinspirasi LLM2Vec, menarik jika causal-to-bidirectional
menjadi bottleneck. Implementasi harus mencegah kebocoran target, termasuk
pergeseran prediksi token yang benar. Contrastive pooling bukan prioritas untuk
exact source copying.

Distillation dari IT decoder yang melihat raw source juga kandidat: samakan
distribusi next-token student dengan teacher pada prefix target yang sama.
Teacher hanya melihat sumber dan token target sebelumnya, bukan future gold.
Biaya teacher dan tambahan paparan data harus dihitung. Mulai dengan CE tugas
yang sederhana; KD bukan komponen wajib dan teacher yang salah tidak membantu.

Hindari MSE hidden-state mentah antar-layer/model tanpa pemetaan posisi dan
fitur yang bermakna. Norm dan dimensi yang sama tidak berarti target representasi
yang sama. Ini alternatif terkontrol, bukan alasan menambah loss sebanyak mungkin.

### Langkah 4: gate kelulusan sebelum full run

Catat pada heldout: exact-copy accuracy, identifier exact match, localization,
patch format/apply/functional success, NLL per tugas, dan correct/empty/swapped
context pada input decoder serta target identik. Tambahkan mass attention encoder,
gradient/update norm per grup, throughput dan peak allocated VRAM.

Mass encoder tinggi tidak sama dengan penggunaan yang benar; mass rendah juga
belum otomatis berarti tidak berguna. Penurunan NLL saja tidak cukup. Kurva
correct-context dan generasi bebas harus menunjukkan penggunaan sumber yang
lebih tepat. Cek kontrol identifier baru agar tidak hanya belajar patch prior.

Tentukan kriteria penerimaan sebelum melihat hasil. Contoh gate rekayasa adalah
copy exact-match >=95% pada tugas copy pendek yang sengaja mudah, jauh di atas
empty/swapped context. Angka ini usulan internal, bukan standar paper; tidak
membuktikan repair menang. Jika gagal copy mudah, hentikan perluasan ke empat
pasangan dan diagnosis jalurnya dahulu.

Sesudah lolos, lakukan SFT decoder dengan anggaran N yang dipin. Evaluasi repair
pada benchmark baru memakai paired bootstrap confidence interval terhadap
baseline. Klaim lintas seed memerlukan pengulangan seed. Jangan memakai benchmark
test untuk memilih schedule pilot.

## 11. Pilihan yang sengaja bukan prioritas pertama

- Mengulang enam training penuh hanya dengan epoch lebih banyak.
- Mengganti optimizer karena AF-Muon, tanpa menguji jalur informasi.
- Mengganti norm, cross RoPE, rank, backbone IT dan objective secara bersamaan.
- Memaksa kompresi source ke sedikit latent query sebelum exact copying terbukti.
- Menduplikasi source ke decoder lalu menyebut kenaikannya bukti encoder-only.
- Memasang cross-attention hanya di global layers sebagai optimisasi gratis:
  T5Gemma 2 melaporkan penurunan kualitas untuk ablasinya.
- Menyamakan memory reserved dengan memori tensor aktif atau menjanjikan speedup
  freeze tanpa profiling.

Model resmi [T5Gemma 2 4B-4B](https://huggingface.co/google/t5gemma-2-4b-4b)
dapat menjadi kontrol tambahan untuk pipeline SFT encoder-decoder yang sudah
diadaptasi. Checkpoint tersebut bukan pengganti langsung Gemma 4 IT; keluarga,
post-training, parameter dan antarmukanya berbeda. Gunakan sebagai kontrol
kelayakan pipeline, bukan baseline utama yang dianggap setara.

## 12. Keputusan praktis

Pertahankan bukti/checkpoint lama. Perubahan berikutnya sebaiknya dimulai dari
**pilot alignment bertahap dengan decoder frozen, objective source fidelity,
dan pembanding direct-joint pada budget pilot sama**. Pertahankan merged pada
percobaan pertama supaya hasil menguji perubahan metode training secara jelas.
Uji encoder IT sebagai satu faktor setelah itu. Cross residual bergate menjadi
alternatif berikutnya bila diagnosis menunjukkan kompetisi joint softmax tetap
menjadi hambatan.

Kemenangan yang paling masuk akal untuk ditargetkan adalah manfaat terukur pada
repair dengan konteks relevan/distraktor/infilling, atau trade-off kualitas dan
biaya inference. Bidirectional mempunyai potensi keuntungan di sana; ia tidak
menjamin kemenangan pada semua repair pendek atas decoder IT yang sudah kuat.
