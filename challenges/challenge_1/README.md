# Challenge 1 — Implementasi Algoritma FedAvg

**Bobot**: 20 poin  
**Tipe**: Kode + Analisis

---

## Latar Belakang

Federated Averaging (FedAvg) adalah algoritma inti dalam Federated Learning yang
diusulkan oleh McMahan et al. (2017). Ide utamanya: daripada mengirim data mentah
ke server, setiap client hanya mengirim **bobot model** yang sudah dilatih secara lokal.
Server kemudian melakukan **rata-rata berbobot** dari semua model lokal.

**Formula FedAvg:**

$$\theta_{global}^{(t+1)} = \sum_{k=1}^{K} \frac{n_k}{N} \cdot \theta_k^{(t)}$$

Di mana:
- $K$ = jumlah client
- $n_k$ = jumlah data client $k$
- $N = \sum_k n_k$ = total data semua client
- $\theta_k^{(t)}$ = parameter model client $k$ setelah round $t$

---

## Tugas

### Bagian A — Implementasi FedAvg (15 poin)

Lengkapi fungsi `fedavg_aggregate()` di file `challenge_1/solution.py`.

Ketentuan:
1. Implementasi harus menggunakan **rata-rata berbobot** (bukan rata-rata sederhana)
2. Bobot adalah proporsi jumlah data tiap client: $w_k = n_k / N$
3. Jangan menggunakan fungsi `fedavg_manual` dari `utils/model_utils.py` — implementasi sendiri!

### Bagian B — Analisis (5 poin)

Jawab pertanyaan berikut di file `challenge_1/analisis.md`:

1. Apa perbedaan antara **rata-rata berbobot** (FedAvg) dan **rata-rata sederhana**?
   Kapan keduanya menghasilkan hasil yang sama?

2. Jika client-1 memiliki 2000 sampel dan client-2 memiliki 500 sampel,
   berapa bobot masing-masing dalam FedAvg?

3. Mengapa FedAvg lebih disukai daripada mengirim gradien (FedSGD)?
   Sebutkan minimal 2 alasan.

---

## File yang Perlu Dimodifikasi

- `challenges/challenge_1/solution.py` ← isi bagian `# TODO`

---

## Cara Menjalankan Test

```bash
cd challenges/challenge_1
python test_fedavg.py
```

Output yang diharapkan:
```
[TEST 1] FedAvg dengan bobot seimbang... PASS
[TEST 2] FedAvg dengan bobot tidak seimbang... PASS  
[TEST 3] FedAvg dengan 4 client... PASS
[TEST 4] Konsistensi dimensi output... PASS
Semua test lulus!
```

---

## Petunjuk

- Parameter model Logistic Regression terdiri dari `coef` (matrix) dan `intercept` (vektor)
- Setiap client mengirim pasangan `(coef_k, intercept_k, n_samples_k)`
- Agregasi dilakukan terpisah untuk `coef` dan `intercept`
- Gunakan `numpy` untuk operasi array

---

## Referensi

- McMahan, H. B., et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS 2017.
- [Paper PDF](https://arxiv.org/pdf/1602.05629.pdf)
