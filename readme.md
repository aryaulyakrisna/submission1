# Laporan Proyek Machine Learning - Prediksi Harga Rumah di Tebet

## Domain Proyek

Harga properti, khususnya rumah di wilayah Tebet, Jakarta Selatan, sangat dipengaruhi oleh berbagai faktor seperti luas tanah, luas bangunan, jumlah kamar tidur, jumlah kamar mandi, dan jumlah garasi. Tanpa adanya sistem prediksi yang akurat, konsumen seperti calon pembeli atau penjual sering kali kesulitan menentukan harga wajar sebuah properti. Hal ini dapat menyebabkan keputusan yang kurang tepat, baik dari segi finansial maupun strategis. Prediksi harga rumah menjadi penting untuk membantu pemangku kepentingan seperti pembeli, penjual, agen properti, dan pengembang dalam membuat keputusan yang lebih terinformasi dan cerdas.

Masalah ini harus diselesaikan karena harga properti yang tidak transparan dapat menghambat transaksi di pasar properti, meningkatkan risiko kerugian finansial, dan mempersulit perencanaan investasi. Dengan pendekatan machine learning, model prediksi dapat memberikan estimasi harga yang objektif berdasarkan fitur-fitur properti, sehingga meningkatkan efisiensi dan kepercayaan di pasar properti. Penelitian sebelumnya menunjukkan bahwa algoritma seperti regresi linier, k-nearest neighbors (KNN), dan boosting seperti AdaBoost dapat digunakan untuk memprediksi harga rumah dengan akurasi yang bervariasi, di mana algoritma berbasis ensemble sering kali lebih unggul untuk data kompleks [1].

**Referensi**:  
[1] I. Kurniawan, N. Rahaningsih, and T. Suprapti, “Implementasi Algoritma Regresi Linier dan K-Nearest Neighbor untuk Prediksi Harga Rumah,” *JATI (Jurnal Mahasiswa Teknik Informatika)*, vol. 8, no. 1, pp. 1187–1193, Feb. 2024.

## Business Understanding

### Problem Statements
- Fitur apa saja yang paling memengaruhi harga rumah di wilayah Tebet?
- Berapa estimasi harga rumah berdasarkan kombinasi fitur tertentu seperti luas tanah, luas bangunan, jumlah kamar tidur, jumlah kamar mandi, dan jumlah garasi?

### Goals
- Mengidentifikasi fitur-fitur yang memiliki pengaruh signifikan terhadap harga rumah di Tebet.
- Membangun model machine learning yang dapat memprediksi harga rumah dengan akurasi tinggi berdasarkan fitur-fitur yang diberikan.

### Solution Statements
Untuk mencapai tujuan di atas, tiga pendekatan solusi diusulkan:
1. **Menggunakan Random Forest Regressor**: Algoritma ini dipilih karena kemampuannya menangani data non-linear dan hubungan kompleks antar fitur. Model akan dioptimasi dengan *hyperparameter tuning* menggunakan *RandomizedSearchCV* untuk meningkatkan akurasi prediksi.
2. **Menggunakan K-Nearest Neighbors (KNN) Regressor**: Algoritma ini dipilih sebagai pembanding karena sifatnya yang sederhana dan efektif untuk data dengan pola distribusi lokal. KNN akan dioptimasi dengan mencari jumlah tetangga (*k*) yang optimal.
3. **Menggunakan AdaBoostRegressor**: Algoritma boosting ini dipilih karena kemampuannya meningkatkan performa model lemah (weak learner) seperti pohon keputusan dengan memberikan bobot lebih pada data yang sulit diprediksi. Model akan dioptimasi untuk menentukan jumlah estimator dan learning rate yang optimal.

Kinerja ketiga model akan diukur menggunakan metrik evaluasi **Mean Squared Error (MSE)**, yang sesuai untuk masalah regresi karena mengukur rata-rata kuadrat selisih antara prediksi dan nilai sebenarnya.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah data daftar harga rumah di wilayah Tebet, Jakarta Selatan, yang bersumber dari Kaggle. Dataset ini dapat diunduh melalui tautan berikut: [Daftar Harga Rumah](https://www.kaggle.com/datasets/wisnuanggara/daftar-harga-rumah). Dataset terdiri dari 1010 baris data dengan 8 kolom, yang mencakup informasi tentang harga rumah dan fitur-fiturnya. Data tersedia dalam format Excel dan tidak mengandung nilai *null*, tetapi terdapat beberapa anomali seperti duplikasi dan harga yang tidak wajar (misalnya, harga di bawah Rp100 juta).

### Variabel-variabel pada Dataset
- **NO**: Nomor urut rumah (tidak digunakan dalam analisis).
- **NAMA RUMAH**: Deskripsi nama atau lokasi rumah (tidak digunakan dalam analisis).
- **HARGA**: Harga rumah dalam Rupiah (variabel dependen/target).
- **LB**: Luas bangunan dalam meter persegi.
- **LT**: Luas tanah dalam meter persegi.
- **KT**: Jumlah kamar tidur.
- **KM**: Jumlah kamar mandi.
- **GRS**: Jumlah garasi.

**Exploratory Data Analysis (EDA)**:

![Oulier](/img/outlier.png)

- Visualisasi distribusi harga menunjukkan bahwa sebagian besar rumah memiliki harga di bawah Rp10 miliar, dengan beberapa *outlier* di atas Rp20 miliar.

![Korelasi](/img/korelasi_antar_fitur.png)

- Korelasi antar fitur dianalisis menggunakan *heatmap*. Luas tanah (LT) dan luas bangunan (LB) memiliki korelasi positif yang kuat dengan harga (masing-masing 0,61 dan 0,75), sedangkan jumlah garasi (GRS) memiliki korelasi yang lebih rendah (0,40).

![Scatter Plot](/img//scatterplot_data.png)

- Scatter plot antara luas tanah dan harga menunjukkan hubungan yang hampir linier untuk rumah dengan luas tanah di bawah 500 m².

## Data Preparation

Tahapan persiapan data dilakukan untuk memastikan dataset siap digunakan dalam pemodelan. Berikut adalah langkah-langkah yang dilakukan:

1. **Seleksi Fitur**:
   - Kolom **NO** dan **NAMA RUMAH** dihapus karena tidak relevan untuk prediksi.
   - Fitur yang digunakan: **LB**, **LT**, **KT**, **KM**, dan **GRS**.

2. **Pembersihan Data**:
   - ![Delete Outlier](/img/delete_outlier.png)
   -
   - Menghapus baris dengan outlier pada LT dan LB (Dari hasil korelasi Luas Tanah dan Luas Bangunan memiliki korelasi tertinggi dengan Harga Tanah).

3. **Transformasi Data**:

   - ``` python
      y = df_filtered["HARGA"]/10000000000 # perkecil dibagi 1000000000
     ```

   - Harga rumah (HARGA) dinormalisasi dengan membaginya dengan 10^10 untuk mempermudah pelatihan model.

   - ![Standarisasi](/img/stendarisasi_standarscaler.png)

   - Fitur numerik (**LB**, **LT**, **KT**, **KM**, **GRS**) diskalakan menggunakan **StandardScaler** untuk memastikan semua fitur berada pada skala yang sama, yang penting untuk algoritma seperti KNN dan AdaBoost.

4. **Pemisahan Data**:

   - ![Train Test Split](/img/bagi_data_traintestsplit.png)
   
   - Data dibagi menjadi data pelatihan (80%) dan data pengujian (20%) menggunakan `train_test_split` dengan *random_state=55* untuk konsistensi.

**Alasan Data Preparation**:
- **Seleksi fitur** dilakukan untuk fokus pada variabel yang relevan dan mengurangi kompleksitas model.
- **Pembersihan data** diperlukan untuk menghilangkan anomali yang dapat mengganggu akurasi prediksi.
- **Normalisasi dan scaling** penting untuk memastikan model tidak bias terhadap fitur dengan skala besar (misalnya, luas tanah vs jumlah garasi), terutama untuk KNN dan AdaBoost yang sensitif terhadap skala.
- **Pemisahan data** memungkinkan evaluasi model pada data yang tidak dilihat selama pelatihan, sehingga mengukur generalisasi model.

## Modeling

Tiga model machine learning digunakan untuk memprediksi harga rumah: **Random Forest Regressor**, **K-Nearest Neighbors (KNN) Regressor**, dan **AdaBoostRegressor**. Berikut adalah tahapan dan parameter yang digunakan:

### 1. Random Forest Regressor (Model Terbaik)
- **Tahapan**:
  - Inisialisasi model dengan parameter awal: `n_estimators=150`, `max_depth=12`, `min_samples_split=5`, `min_samples_leaf=3`, `random_state=55`, `n_jobs=-1`.
  - Optimasi hyperparameter menggunakan `RandomizedSearchCV` dengan parameter berikut:
    ```python
    param_dist = {
        'n_estimators': [100, 150, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 3, 5],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    ```
  - Model terbaik: `RandomForestRegressor(max_depth=50, max_features='sqrt', n_estimators=300, n_jobs=-1, random_state=55)`.

- **Kelebihan**:
  - Menangani hubungan non-linear dan interaksi antar fitur dengan baik.
  - Robust terhadap *outlier* dan data yang tidak seimbang.
- **Kekurangan**:
  - Membutuhkan waktu komputasi lebih lama untuk data besar.
  - Sulit diinterpretasikan dibandingkan model linier.

### 2. K-Nearest Neighbors (KNN) Regressor
- **Tahapan**:
  - Inisialisasi model dengan parameter awal: `n_neighbors=5`.
  - Optimasi hyperparameter untuk menemukan jumlah tetangga (*k*) optimal menggunakan validasi silang dengan rentang `k` dari 1 hingga 20.
- **Kelebihan**:
  - Sederhana dan intuitif, cocok untuk data dengan pola lokal.
  - Tidak memerlukan asumsi distribusi data.
- **Kekurangan**:
  - Sensitif terhadap skala data, sehingga memerlukan *scaling*.
  - Performa menurun pada data dengan dimensi tinggi atau jumlah data besar.

### 3. AdaBoostRegressor
- **Tahapan**:
  - Inisialisasi model dengan *base estimator* berupa pohon keputusan (`DecisionTreeRegressor`) dengan kedalaman maksimum 3.
  - Parameter awal: `n_estimators=50`, `learning_rate=0.005`, `random_state=55`.
- **Kelebihan**:
  - Meningkatkan performa model lemah dengan fokus pada data yang sulit diprediksi.
  - Relatif robust terhadap *overfitting* jika *learning rate* diatur dengan baik.
- **Kekurangan**:
  - Sensitif terhadap *outlier* karena memberikan bobot lebih pada data yang salah diprediksi.
  - Membutuhkan waktu lebih lama untuk pelatihan dibandingkan KNN.

### Model Terbaik
Random Forest Regressor dipilih sebagai model terbaik berdasarkan hasil evaluasi (setelah tuning):
- **Train MSE**: 0.0121
- **Test MSE**: 0.0531

**Perbandingan dengan Model Lain**:
- **KNN Regressor**:
  - **Train MSE**: 	0.044523	
  - **Test MSE**: 0.046944
  - Performa lebih buruk karena sensitivitas terhadap *outlier* dan kebutuhan data yang lebih homogen.
- **AdaBoostRegressor**:
  - **Train MSE**: 0.075093
  - **Test MSE**: 0.076101
  - Performa lebih baik dari KNN tetapi masih di bawah Random Forest karena sensitivitas terhadap *outlier* di dataset.

Random Forest lebih robust dan mampu menangani variasi dalam data properti Tebet, sementara AdaBoost memberikan hasil yang kompetitif tetapi kurang optimal untuk data dengan *outlier*. KNN kurang cocok karena ketergantungannya pada distribusi data lokal.

**Proses Improvement**:
- Hyperparameter tuning pada Random Forest mengurangi *overfitting* dengan membatasi `max_depth` dan meningkatkan jumlah pohon (`n_estimators`).
- Untuk AdaBoost, pengaturan *learning rate* rendah (0.1) dan jumlah estimator tinggi (200) membantu meningkatkan generalisasi.

## Evaluation

Metrik evaluasi yang digunakan adalah **Mean Squared Error (MSE)**, yang dihitung sebagai:

- **Penjelasan Metrik**:
  - MSE mengukur rata-rata kuadrat selisih antara harga prediksi (\(\hat{y}_i\)) dan harga sebenarnya (\(y_i\)).
  - Nilai MSE yang lebih kecil menunjukkan prediksi yang lebih akurat.
  - MSE dipilih karena sesuai untuk masalah regresi dan sensitif terhadap kesalahan besar, yang penting dalam konteks harga rumah di mana kesalahan besar dapat berdampak finansial signifikan.

- **Hasil Proyek**:
  - **Random Forest Regressor**:
    - **Train MSE**: 0.0121
    - **Test MSE**: 0.0531

  - ![Prediksi Data](/img/predict_data.png)

  - Contoh prediksi (Random Forest): Untuk rumah dengan luas bangunan 220 m², luas tanah 220 m², 3 kamar tidur, 3 kamar mandi, dan 0 garasi, model memprediksi harga sekitar Rp4,12 miliar, yang sesuai dengan kisaran harga pasar di Tebet.
  - Fitur paling berpengaruh adalah luas tanah (LT) dan luas bangunan (LB), yang konsisten dengan hasil EDA.

**Penjelasan Kinerja**:
- Random Forest memberikan MSE terendah pada data pengujian, menunjukkan akurasi dan generalisasi yang lebih baik dibandingkan KNN dan AdaBoost.
- AdaBoost mengungguli KNN tetapi masih kalah dari Random Forest karena sensitivitas terhadap *outlier* di dataset.
- Selisih antara Train MSE dan Test MSE pada Random Forest menunjukkan sedikit *overfitting*, tetapi masih dalam batas wajar untuk aplikasi praktis.
- Model Random Forest dapat digunakan oleh agen properti atau pembeli untuk memperkirakan harga rumah dengan cepat berdasarkan fitur-fitur utama.
