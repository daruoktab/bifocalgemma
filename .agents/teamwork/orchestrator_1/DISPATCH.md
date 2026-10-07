# Dispatch Log

## 2026-10-07T09:45:47Z
You are the Project Orchestrator (teamwork_preview_orchestrator) for the following project.

Your assigned working directory is:
d:\Codings\bifocalgemma\.agents\teamwork\orchestrator_1

The original user request is recorded in:
d:\Codings\bifocalgemma\.agents\teamwork\ORIGINAL_REQUEST.md

Project root:
d:\Codings\bifocalgemma

Context and Task:
Cari dan evaluasi dataset tambahan untuk SFT BifocalGemma (instruction-to-patch, code editing/repair Python) yang selaras dengan benchmark evaluasi repo ini (HumanEvalFix/HumanEvalPack, CanItEdit, CrossCodeEval), dengan tujuan menaikkan skor benchmark tanpa kontaminasi. Hasilnya berupa laporan riset terverifikasi; TIDAK ADA perubahan pada kode notebook/sumber (src/, tools/, tests/, notebooks).

Konteks Repo:
- Dataset saat ini: nuprl/EditPackFT (default), bigcode/commitpackft (python), local_jsonl.
- Benchmark evaluasi (docs/DATASETS_AND_BENCHMARKS.md): HumanEvalFix/HumanEvalPack, CanItEdit, CrossCodeEval.
- Kontrak data saat ini: input = instruksi + kode sebelum edit; target = unified diff; split grouped per commit; tanpa RL/VM/Docker.
- Hardware target: satu GPU Molab. Murni offline SFT (instruction + pre-code + post-code/patch). Tanpa RL, VM, Docker, atau reward/teacher model.

Requirements & Acceptance Criteria:
1. R1: Minimal 5 kandidat dataset baru dievaluasi (di luar EditPackFT dan CommitPackFT). Murni offline SFT.
2. R2: Verifikasi dari sumber aslinya (API/halaman resmi, inspect repo/huggingface): lisensi, jumlah record, field yang tersedia, revision yang bisa dipin, dan pemetaan ke kontrak data repo. Nilai relevansi terhadap HumanEvalFix, CanItEdit, CrossCodeEval. Independent verification: pastikan minimal 3 kandidat diverifikasi silang/independen terhadap sumber asli.
3. R3: Cek overlap dan kontaminasi terhadap ketiga benchmark (HumanEvalFix, CanItEdit, CrossCodeEval) secara terukur (hash / near-duplicate / analisis repo & commit). Dataset yang berisiko overlap dilaporkan terpisah dan TIDAK boleh masuk rekomendasi utama.
4. R4: Laporan terstruktur berisi: kandidat berperingkat, bukti verifikasi, hasil cek overlap, estimasi ukuran & beban (disk, waktu training, VRAM), rekomendasi urutan pemakaian dan cara pencampuran dengan data saat ini.
5. Kejujuran klaim: jangan menjanjikan kenaikan skor tanpa bukti empiris/literatur (tandai sebagai hipotesis).
6. ZERO CODE MODIFICATIONS: Pastikan git status tidak ada perubahan pada notebook, src/, tools/, tests/. Semua laporan dan artifact koordinasi diletakkan di folder riset/dokumen yang sesuai (misal docs/ atau laporan terstruktur yang ditentukan).
