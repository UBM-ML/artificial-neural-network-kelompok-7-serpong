# Laporan Kelompok — ANN Bake-Off

## Identitas Kelompok

- **Nama Kelompok:** Kelompok 7 Serpong
- **Anggota:**
  1. 32220122 - Andrianno Thio Brawijaya
  2. 32230008 - Kelvin Ferlysanto
  3. 32230015 - Albert Aptryhas - Varian 04_mlp_relu
  4. 32230029 - Melvin Wijaya Susanto - Varian 01_single_layer
  5. 32230032 - Charles Hartanto - Gabungan + report + eksperimen & analisis akurasi pada hidden layer yang ditambahkan
  6. 32230039 - Justin Wijaya - Varian 02_mlp_sigmoid
  7. 32230040 - Dhevent Orlando - Varian 03_mlp_tanh

---

## 1. Ringkasan Hasil Eksperimen

Kami melatih empat varian Neural Network pada dataset Iris (150 sampel, 4 fitur, 3 kelas: setosa, versicolor, virginica). Semua varian menggunakan konfigurasi identik: 100 epoch, batch size 8, optimizer Adam, loss categorical_crossentropy, random seed 42. Satu-satunya variabel yang berbeda adalah arsitektur dan fungsi aktivasi hidden layer.

<img width="1005" height="509" alt="image" src="https://github.com/user-attachments/assets/c6eaf39f-13f5-4c7b-ade8-08c82871dd4a" />

| Varian | Arsitektur | Aktivasi | Val Accuracy | Val Loss | Train Accuracy | Train Loss |
|--------|-----------|----------|:------------:|:--------:|:--------------:|:----------:|
| 01_single_layer | Input(4) -> Output(3) | — | 0.7500 | 0.5437 | 0.8646 | 0.4167 |
| 02_mlp_sigmoid | Input(4) -> Dense(16) -> Output(3) | Sigmoid | 0.9583 | 0.3086 | 0.9167 | 0.2511 |
| 03_mlp_tanh | Input(4) -> Dense(16) -> Output(3) | Tanh | 0.9583 | 0.1265 | 0.9688 | 0.1138 |
| 04_mlp_relu | Input(4) -> Dense(16) -> Output(3) | ReLU | 0.9583 | 0.0807 | 0.9896 | 0.0346 |

---

## 2. Analisis & Diskusi

### 2.1 Apakah single-layer mampu mencapai akurasi yang sebanding dengan multi-layer? Mengapa?

Tidak. Single-layer hanya mencapai val_accuracy 75.00% di epoch 100, sementara ketiga varian MLP mencapai 95.83%. Selisihnya 20.83 poin persentase, jauh dari sebanding.

Penyebabnya bersifat struktural. Single-layer tidak memiliki hidden layer, sehingga transformasi yang bisa dilakukan hanyalah linear: input dikalikan bobot lalu langsung masuk softmax. Model ini setara dengan Multinomial Logistic Regression. Dataset Iris memiliki kelas versicolor dan virginica yang tidak linear-separable di ruang fitur asli, sehingga batas keputusan linear tidak cukup untuk memisahkan keduanya.

Bukti dari data: di 10 epoch pertama, train accuracy single-layer hanya 21-28% dan val_accuracy 12-16%. Model sangat lambat menangkap pola karena memang tidak punya kapasitas non-linear. Bahkan setelah 100 epoch, train accuracy-nya (86.46%) masih di bawah val_accuracy MLP mana pun. Gap train-val sebesar +11.46% juga menunjukkan model tidak konsisten antara training dan validation.

### 2.2 Apakah ReLU benar-benar konvergen lebih cepat dibanding Sigmoid? Buktikan dengan data.

Ya, terbukti secara empiris. Berikut perbandingan epoch pertama kali val_accuracy menembus 90%:

| Varian | Epoch Pertama Val Acc >= 90% | Peak Val Acc Sepanjang Training |
|--------|:----------------------------:|:-------------------------------:|
| 04_mlp_relu | **Epoch 14** | 100.00% (epoch 21) |
| 03_mlp_tanh | Epoch 24 | 95.83% (epoch 47) |
| 02_mlp_sigmoid | Epoch 34 | 95.83% (epoch 96) |
| 01_single_layer | Tidak pernah | 75.00% (epoch 73) |

ReLU 20 epoch lebih cepat dari Sigmoid untuk mencapai threshold yang sama. Selain itu, ReLU mencapai val_accuracy 100% pada epoch 21, sesuatu yang tidak pernah dicapai Sigmoid atau Tanh sepanjang 100 epoch training.

Dari sisi loss, train loss ReLU di epoch 100 adalah 0.0346 dibanding Sigmoid 0.2511. ReLU 7x lebih efisien dalam menekan loss pada jumlah epoch yang sama. Alasannya: ReLU tidak mengalami vanishing gradient untuk nilai positif karena gradiennya konstan sebesar 1. Sigmoid saturasi di ujung kurva sehingga gradiennya mendekati nol dan update bobot menjadi sangat kecil di awal training.

### 2.3 Jika kelompok menambah hidden layer ke 3 atau 4, apakah akurasi terus naik? Lakukan eksperimen tambahan.

Tidak. Eksperimen tambahan membuktikan bahwa menambah hidden layer tidak selalu meningkatkan akurasi. Berikut hasil empirisnya (semua menggunakan ReLU, 16 unit per layer, konfigurasi identik dengan varian utama):

| Jumlah Hidden Layer | Test Accuracy | Test Loss | Val Accuracy | Jumlah Parameter |
|:-------------------:|:-------------:|:---------:|:------------:|:----------------:|
| 1 | 0.9333 | 0.1505 | 0.9583 | 131 |
| 2 | **0.9667** | **0.0714** | 0.9583 | 403 |
| 3 | 0.9333 | 0.1223 | 0.9583 | 675 |
| 4 | 0.8667 | 0.2434 | 0.9583 | 947 |

Tiga temuan utama dari data ini:

**Pertama**, test accuracy memuncak di 2 hidden layer (96.67%) lalu turun di layer ke-3 (93.33%) dan ke-4 (86.67%). Menambah layer justru menurunkan performa pada dataset kecil ini.

**Kedua**, val_accuracy keempat konfigurasi sama persis di 95.83%. Ini bukan berarti performanya identik, melainkan karena validation set hanya 30 sampel sehingga tidak cukup sensitif untuk membedakan model. Test accuracy memberikan gambaran yang lebih nyata.

**Ketiga**, test loss model 4 layer (0.2434) jauh lebih tinggi dari model 2 layer (0.0714). Ini tanda bahwa model 4 layer mulai kesulitan belajar dengan baik, kemungkinan karena vanishing gradient yang makin parah seiring bertambahnya layer dengan dataset yang sangat kecil (hanya 120 sampel training).

---

## 3. Refleksi Proses Kerja Kelompok

Kami dari kelompok 7 Serpong membagi tugas berdasarkan varian model. Masing-masing anggota bertanggung jawab atas satu varian: Melvin mengerjakan 01_single_layer, Justin mengerjakan 02_mlp_sigmoid, Dhevent mengerjakan 03_mlp_tanh, dan Albert mengerjakan 04_mlp_relu. Charles bertugas untuk menggabungkan hasil keempat varian di notebook 05_comparison dan menyusun laporan ini.

Pembagian ini mengikuti pola controlled experiment: setiap orang berfokus pada satu variabel (fungsi aktivasi) dengan semua parameter lain dikunci lewat file `config.py` dan `data_loader.py` yang sama. Pendekatan ini memastikan perbandingan antar varian bersifat adil dan hasilnya dapat dipercaya.

Kesulitan teknis yang muncul adalah koordinasi format file CSV hasil training. Di awal, beberapa anggota menyimpan history dengan nama kolom yang sedikit berbeda, sehingga notebook comparison tidak bisa membaca semua file sekaligus.

Kesulitan non-teknis yang muncul adalah sinkronisasi waktu pengerjaan. Karena notebook comparison bergantung pada output keempat varian, Charles tidak bisa mulai menggabungkan sebelum semua anggota selesai.

Pelajaran yang bisa dibawa ke proyek ML berikutnya ada tiga. Pertama, tentukan format output sejak awal sebelum mulai coding, bukan setelah semua orang selesai. Kedua, gunakan shared config file untuk mengunci semua parameter yang tidak boleh berbeda antar eksperimen. Ketiga, dalam controlled experiment, satu perubahan variabel saja sudah cukup untuk menghasilkan insight yang bermakna dan tidak perlu mengubah banyak hal sekaligus.

---

## 4. Kontribusi Tiap Anggota

| Anggota | Kontribusi Konkret | % Effort |
|---------|-------------------|:--------:|
| Andrianno Thio Brawijaya | | |
| Kelvin Ferlysanto | | |
| Albert Aptryhas | Implementasi dan training varian 04_mlp_relu| 20% |
| Melvin Wijaya Susanto | Implementasi dan training varian 01_single_layer| 20% |
| Charles Hartanto | Menggabungkan semua hasil di 05_comparison.ipynb, menyusun REPORT.md, menjalankan eksperimen tambahan hidden layer (1-4 layer) dan menganalisis hasilnya | 20% |
| Justin Wijaya | Implementasi dan training varian 02_mlp_sigmoid| 20% |
| Dhevent Orlando | Implementasi dan training varian 03_mlp_tanh| 20% |

Total: 100%

---

## 5. Referensi
