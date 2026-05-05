# Final Project: Federated Learning pada Mobile Activity Recognition

**Mata Kuliah**: Sistem Komputasi Terdistribusi  
**Topik**: Federated Learning — Mobile Sensor Data (UCI HAR Dataset)  
**Infrastruktur**: 1 VM Server (Global Model) + 4 VM Client (Mahasiswa)

---

## Deskripsi Proyek

Proyek ini mensimulasikan sistem **Federated Learning (FL)** secara nyata menggunakan 5 Virtual Machine:

| VM | Peran | Keterangan |
|----|-------|-----------|
| `vm-server` | FL Server | Agregasi global model (FedAvg) |
| `vm-client-1` | FL Client 1 | Melatih model lokal, data partisi 1 |
| `vm-client-2` | FL Client 2 | Melatih model lokal, data partisi 2 |
| `vm-client-3` | FL Client 3 | Melatih model lokal, data partisi 3 |
| `vm-client-4` | FL Client 4 | Melatih model lokal, data partisi 4 |

### Skenario

Bayangkan 4 pengguna smartphone yang masing-masing menyimpan data aktivitas fisik mereka secara **lokal**. Data ini tidak boleh dikirim ke server (privasi). FL memungkinkan keempat pengguna tersebut berkolaborasi melatih model klasifikasi aktivitas **tanpa berbagi data mentah**.

---

## Dataset: UCI HAR (Human Activity Recognition)

- **Sumber**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- **Sensor**: Accelerometer + Gyroscope dari smartphone Samsung Galaxy S II
- **Fitur**: 561 fitur (domain waktu dan frekuensi)
- **Label**: 6 kelas aktivitas:
  | Kode | Aktivitas |
  |------|-----------|
  | 1 | WALKING |
  | 2 | WALKING_UPSTAIRS |
  | 3 | WALKING_DOWNSTAIRS |
  | 4 | SITTING |
  | 5 | STANDING |
  | 6 | LAYING |
- **Ukuran**: 7.352 training samples, 2.947 test samples

---

## Arsitektur Sistem

```
  ┌─────────────────────────────────────────────┐
  │              VM SERVER (Global Model)       │
  │                                             │
  │  ┌─────────────────────────────────────┐    │
  │  │  Flask REST API  (port 5000)        │    │
  │  │  - GET  /api/model    → kirim model │    │
  │  │  - POST /api/update   → terima upd  │    │
  │  │  - GET  /api/status   → cek status  │    │
  │  │  - GET  /api/results  → lihat hasil │    │
  │  └─────────────────────────────────────┘    │
  │                    ▲                        │
  │              FedAvg Aggregation             │
  └────────┬─────────┬──────────┬──────────┬────┘
           │         │          │          │
     HTTP  │   HTTP  │    HTTP  │    HTTP  │
           ▼         ▼          ▼          ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ CLIENT 1 │ │ CLIENT 2 │ │ CLIENT 3 │ │ CLIENT 4 │
  │          │ │          │ │          │ │          │
  │ Data P1  │ │ Data P2  │ │ Data P3  │ │ Data P4  │
  │ Train LR │ │ Train LR │ │ Train LR │ │ Train LR │
  └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

**Algoritma FedAvg** (McMahan et al., 2017):
$$\theta_{global} = \sum_{k=1}^{K} \frac{n_k}{N} \cdot \theta_k$$

Di mana $n_k$ = jumlah data client $k$, $N$ = total data semua client.

---

## Struktur Direktori

```
case-2-fl-mobile/
├── README.md                    ← Panduan ini
├── setup.sh                     ← Script instalasi otomatis
│
├── server/
│   ├── app.py                   ← FL Server (jalankan di vm-server)
│   └── requirements.txt
│
├── client/
│   ├── client.py                ← FL Client (jalankan di vm-client-X)
│   └── requirements.txt
│
├── data/
│   ├── download_data.py         ← Download dataset UCI HAR
│   └── partition_data.py        ← Partisi data ke 4 client
│
├── utils/
│   └── model_utils.py           ← Helper: serialisasi, evaluasi model
│
├── evaluation/
│   └── evaluate_global.py       ← Evaluasi model global
│
└── challenges/
    ├── challenge_1/             ← [WAJIB] Implementasi FedAvg
    ├── challenge_2/             ← [WAJIB] Analisis IID vs Non-IID
    ├── challenge_3/             ← [BONUS] Differential Privacy
    └── challenge_4/             ← [BONUS] Konvergensi & Komunikasi
```

---

## Cara Menjalankan

### Langkah 0 — Persiapan (semua VM)

```bash
# Clone/copy project ke semua VM
cd ~
git clone <repo-url> case-2-fl-mobile
cd case-2-fl-mobile

# Instal dependencies
bash setup.sh
```

### Langkah 1 — Unduh dan Partisi Dataset (jalankan di vm-server)

```bash
# Unduh dataset UCI HAR
python data/download_data.py

# Buat 4 partisi data (IID default)
python data/partition_data.py --mode iid --num_clients 4

# Distribusikan data ke masing-masing client:
# Salin data/partitions/client_1/ ke vm-client-1
# Salin data/partitions/client_2/ ke vm-client-2
# dst.
# Contoh menggunakan scp:
# scp -r data/partitions/client_1 user@<IP_CLIENT_1>:~/case-2-fl-mobile/data/my_partition/
```

### Langkah 2 — Jalankan Server (vm-server)

```bash
# Edit konfigurasi jika perlu
export FL_NUM_CLIENTS=4
export FL_NUM_ROUNDS=10
export FL_SERVER_PORT=5000

python server/app.py
# Output: "FL Server running on http://0.0.0.0:5000"
# Output: "Menunggu 4 client untuk round 1..."
```

### Langkah 3 — Jalankan Client (masing-masing vm-client-X)

```bash
# Di vm-client-1:
export CLIENT_ID=1
export SERVER_URL=http://<IP_SERVER>:5000
export DATA_PATH=data/my_partition/

python client/client.py

# Di vm-client-2:
export CLIENT_ID=2
export SERVER_URL=http://<IP_SERVER>:5000
python client/client.py

# ... dst. untuk client 3 dan 4
```

### Langkah 4 — Monitor Training (vm-server)

```bash
# Cek status di browser atau curl:
curl http://localhost:5000/api/status

# Setelah selesai, lihat hasil:
curl http://localhost:5000/api/results
```

### Langkah 5 — Evaluasi Model Global

```bash
python evaluation/evaluate_global.py --server_url http://<IP_SERVER>:5000
```

---

## Tugas Mahasiswa

### Laporan (40 poin)
1. Jelaskan cara kerja Federated Learning dan perbedaannya dengan centralized ML
2. Analisis hasil eksperimen: akurasi per round, waktu training
3. Diskusikan trade-off: privasi vs akurasi vs komunikasi

### Kode (60 poin)

| Challenge | Topik | Bobot |
|-----------|-------|-------|
| Challenge 1 | Implementasi FedAvg | 20 poin |
| Challenge 2 | Analisis IID vs Non-IID | 20 poin |
| Challenge 3 | Differential Privacy (Bonus) | +10 poin |
| Challenge 4 | Analisis Konvergensi (Bonus) | +10 poin |

Lihat folder `challenges/` untuk detail masing-masing tantangan.

---

## Referensi

- McMahan, H. B., et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS.
- Konečný, J., et al. (2016). *Federated Optimization: Distributed Optimization Beyond the Datacenter*.
- Li, T., et al. (2020). *Federated Learning: Challenges, Methods, and Future Directions*.
- Anguita, D., et al. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN.
