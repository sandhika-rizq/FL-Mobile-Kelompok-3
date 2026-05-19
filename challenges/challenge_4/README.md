# Challenge 4 — Analisis Konvergensi & Efisiensi Komunikasi (BONUS +10 poin)

**Bobot**: +10 poin bonus  
**Tipe**: Eksperimen + Analisis

---

## Latar Belakang

Federated Learning menghadapi tantangan **efisiensi komunikasi** karena:
1. Model harus dikirim bolak-balik antara server dan banyak client setiap round
2. Bandwidth jaringan terbatas, terutama di mobile
3. Terlalu banyak round komunikasi = biaya mahal

### Pertanyaan Kunci:
- Berapa round minimal yang dibutuhkan untuk mencapai akurasi yang baik?
- Apakah lebih banyak epoch lokal lebih efisien?
- Apa yang terjadi jika hanya sebagian client berpartisipasi per round?

---

## Tugas

### Bagian A — Analisis Konvergensi (5 poin)

Implementasikan `analyze_convergence()`:
1. Jalankan FL dengan parameter berbeda:
   - `local_epochs` ∈ {1, 5, 10, 20, 50}
2. Ukur:
   - Akurasi per round
   - Total waktu training
   - Perkiraan total data yang dikomunikasikan (dalam MB)
3. Plot grafik akurasi vs round untuk setiap konfigurasi

### Bagian B — Partial Client Participation (5 poin)

Dalam FL nyata, tidak semua client online setiap round.
Implementasikan `analyze_partial_participation()`:
1. Jalankan FL dengan partial participation: `fraction` ∈ {0.25, 0.5, 0.75, 1.0}
2. Bandingkan konvergensi
3. Analisis: seberapa banyak client minimal agar FL tetap stabil?

---

## Cara Menjalankan

```bash
cd challenges/challenge_4
python solution.py
```

---

## Referensi

- McMahan, H. B., et al. (2017). *Communication-Efficient Learning* (Section 3: FedAvg dengan E dan B).
- Li, T., et al. (2019). *Convergence of FedAvg on Non-IID Data*. ICLR 2020.
