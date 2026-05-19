# Template Analisis — Challenge 1

**Nama Mahasiswa** : Muhammad Zulfikar Raditya Wimbyarto / Michael Andro Nathaniel  
**NIM**            : 235150201111034 / 235150207111039  
**Tanggal**        : 18/05/2026

---

## Pertanyaan 1

**Apa perbedaan antara rata-rata berbobot (FedAvg) dan rata-rata sederhana?
Kapan keduanya menghasilkan hasil yang sama?**

*Jawaban:*

> Rata-rata sederhana memberi kontribusi yang sama untuk setiap client, sedangkan rata-rata berbobot (FedAvg) memberi kontribusi lebih besar kepada client yang memiliki lebih banyak data. Dalam FedAvg setiap model lokal dikalikan dengan bobot `n_samples_k / total_samples` sebelum dijumlahkan.
>
> Keduanya menghasilkan hasil yang sama jika semua client memiliki jumlah data yang sama, sehingga semua bobot `n_samples_k / total_samples` menjadi sama.

---

## Pertanyaan 2

**Jika client-1 memiliki 2000 sampel dan client-2 memiliki 500 sampel,
berapa bobot masing-masing dalam FedAvg?**

*Jawaban:*

> Total sampel = 2000 + 500 = 2500
>
> Bobot client-1 = 2000 / 2500 = 0.8
>
> Bobot client-2 = 500 / 2500 = 0.2

---

## Pertanyaan 3

**Mengapa FedAvg lebih disukai daripada FedSGD?
Sebutkan minimal 2 alasan.**

*Jawaban:*

> FedAvg lebih disukai karena mengurangi beban komunikasi dan menggunakan update model yang sudah dilatih secara lokal sebelum dikirim ke server.
>
> Alasan 1: FedAvg mengirim bobot model/parameter yang sudah diupdate oleh client setelah beberapa iterasi lokal, sehingga jumlah komunikasi menjadi lebih sedikit dibanding mengirim gradien di setiap langkah (FedSGD).
>
> Alasan 2: FedAvg lebih stabil pada data yang tidak seimbang dan terdistribusi heterogen, karena setiap client memberikan kontribusi berdasarkan banyaknya data lokalnya dan pembelajaran lokal dapat mengurangi variansi gradien.

