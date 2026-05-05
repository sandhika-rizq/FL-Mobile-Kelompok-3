# Challenge 3 — Differential Privacy (BONUS +10 poin)

**Bobot**: +10 poin bonus  
**Tipe**: Kode + Analisis

---

## Latar Belakang

**Differential Privacy (DP)** adalah teknik matematis yang memberikan jaminan privasi formal. Dalam FL, DP diterapkan dengan menambahkan **noise Gaussian** ke parameter model sebelum dikirim ke server.

### Mengapa Perlu Differential Privacy?

Meskipun dalam FL data mentah tidak dikirim, model parameter (coef, intercept) masih bisa **membocorkan informasi** tentang data training melalui serangan seperti:
- **Model Inversion Attack**: merekonstruksi data dari gradien
- **Membership Inference Attack**: mengetahui apakah suatu data digunakan dalam training

### Gaussian Mechanism

Untuk menjamin $(ε, δ)$-DP, tambahkan noise Gaussian ke parameter:

$$\tilde{\theta}_k = \theta_k + \mathcal{N}(0, \sigma^2 I)$$

Di mana:
$$\sigma = \frac{\Delta f \cdot \sqrt{2 \ln(1.25/\delta)}}{\varepsilon}$$

- $\Delta f$ = **sensitivity** (L2 norm maksimum parameter)
- $\varepsilon$ = privacy budget (lebih kecil = lebih privat)
- $\delta$ = probabilitas pelanggaran privasi (biasanya $10^{-5}$)

### Trade-off Privasi-Akurasi

```
Privasi ↑ (ε kecil)  →  Noise besar  →  Akurasi ↓
Privasi ↓ (ε besar)  →  Noise kecil  →  Akurasi ↑
```

---

## Tugas

### Bagian A — Implementasi Gaussian Noise (5 poin)

Lengkapi fungsi `add_gaussian_noise()` di `solution.py`.

### Bagian B — Eksperimen Privacy-Utility Trade-off (5 poin)

Lengkapi `experiment_privacy_tradeoff()`:
1. Jalankan FL dengan berbagai nilai epsilon: `[0.1, 0.5, 1.0, 5.0, 10.0, ∞]`
2. Plot grafik akurasi vs epsilon
3. Tentukan nilai epsilon yang memberikan trade-off terbaik

---

## Cara Menjalankan

```bash
cd challenges/challenge_3
python solution.py
```

---

## Referensi

- Dwork, C. & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*.
- Abadi, M., et al. (2016). *Deep Learning with Differential Privacy*. CCS 2016.
- McMahan, H. B., et al. (2018). *Learning Differentially Private Recurrent Language Models*.
