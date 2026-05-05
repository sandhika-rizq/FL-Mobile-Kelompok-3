# Challenge 2 — Analisis IID vs Non-IID Data

**Bobot**: 20 poin  
**Tipe**: Kode + Eksperimen + Analisis

---

## Latar Belakang

Salah satu tantangan terbesar Federated Learning adalah **data Non-IID** (non-independent and identically distributed). Dalam kenyataan, setiap pengguna smartphone memiliki kebiasaan aktivitas yang berbeda:
- Pengguna A lebih banyak berjalan (WALKING)
- Pengguna B lebih banyak duduk (SITTING)
- Pengguna C banyak berbaring (LAYING)

Situasi ini menciptakan **data heterogen** yang memengaruhi konvergensi dan akurasi model global.

### Visualisasi IID vs Non-IID

```
IID (merata):                    Non-IID (miring):
Client 1: ██████████ setiap kelas    Client 1: ████████████████ WALKING
Client 2: ██████████ setiap kelas    Client 2: ████████████ SITTING
Client 3: ██████████ setiap kelas    Client 3: ██████████████ LAYING
Client 4: ██████████ setiap kelas    Client 4: ██████████ STANDING
```

### Distribusi Dirichlet

Non-IID dimodelkan dengan distribusi Dirichlet dengan parameter **alpha (α)**:
- $\alpha \to \infty$: distribusi mendekati IID (merata)  
- $\alpha = 1.0$: heterogen sedang
- $\alpha = 0.1$: sangat heterogen (label skewed)

---

## Tugas

### Bagian A — Eksplorasi Data (5 poin)

Lengkapi fungsi `analyze_distribution()` di `solution.py` untuk:
1. Menghitung distribusi label per client
2. Menghitung **Earth Mover's Distance (EMD)** antar distribusi client
3. Visualisasikan sebagai bar chart (matplotlib)

### Bagian B — Simulasi FL dengan IID & Non-IID (10 poin)

Lengkapi `simulate_federated_learning()` di `solution.py`:
1. Jalankan FL lokal (tanpa server) dengan data IID
2. Jalankan FL lokal dengan data Non-IID (alpha=0.5)
3. Bandingkan kurva akurasi per round
4. Plot grafik perbandingan

### Bagian C — Analisis (5 poin)

Jawab di `challenge_2/analisis.md`:
1. Mengapa akurasi model FL pada data Non-IID lebih rendah / lebih lambat konvergen dibanding IID?
2. Apa yang terjadi jika alpha sangat kecil (misalnya 0.01)?
3. Sebutkan 2 teknik yang dapat meningkatkan performa FL pada Non-IID data.
   Hint: cari tentang **FedProx** atau **SCAFFOLD**.

---

## Cara Menjalankan

```bash
# Pastikan partisi IID sudah ada:
python data/partition_data.py --mode iid --num_clients 4

# Jalankan analisis:
cd challenges/challenge_2
python solution.py

# Output: grafik distribusi + kurva konvergensi
```

---

## Referensi

- Li, T., et al. (2020). *Federated Optimization in Heterogeneous Networks (FedProx)*. MLSys 2020.
- Hsieh, K., et al. (2020). *Quagmire in Non-IID: Empirical Analysis of FL in Distributed Settings*.
- Zhao, Y., et al. (2018). *Federated Learning with Non-IID Data*. arXiv.
