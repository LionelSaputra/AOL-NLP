# Sentiment Analysis NLP Pipeline

Repositori ini berisi implementasi dari saluran pipa (*pipeline*) *Machine Learning* untuk menganalisis sentimen dari ulasan teks (positif/negatif) menggunakan bahasa Python. Proyek ini dibangun dengan struktur yang modular sehingga berbagai model dapat dilatih menggunakan dataset dan metode pembersihan data yang sama.

Dataset utama yang digunakan dalam proyek ini adalah **IMDB Movie Reviews (50.000 data)**.

## 📂 Struktur Proyek
Proyek ini dibagi menjadi tiga bagian utama:
1. `shared_pipeline/` : Modul utama untuk menangani dan membersihkan data.
2. `naive_bayes_classifier-main/` : Implementasi model *Multinomial Naive Bayes*.
3. `logistic_regression_classifier/` : Implementasi model *Logistic Regression*.

---

## 🧹 Data Preprocessing (`shared_pipeline`)
Sebelum teks dimasukkan ke dalam model *Machine Learning*, data mentah dibersihkan menggunakan `data_preprocessing.py`. Proses pembersihan ini meliputi:
1. **Pembersihan Missing Values & Duplikat:** Menghapus baris yang kosong atau memiliki teks yang sama persis.
2. **Lowercasing:** Mengubah semua huruf menjadi huruf kecil.
3. **Contraction Expansion:** Mengembangkan singkatan bahasa Inggris (misalnya `can't` menjadi `cannot`).
4. **Membersihkan Noise:** Menghapus tautan URL, tag HTML (seperti `<br />`), *mentions* (`@`), *hashtags* (`#`), dan angka/tanda baca.
5. **Stopwords Removal:** Menghapus kata-kata penghubung umum yang tidak bermakna (seperti *the, is, in, and*).
6. **Label Encoding:** Mengubah label teks (positif/negatif) menjadi angka yang dipahami komputer (1/0).

---

## 🤖 Model Machine Learning
Kedua model menggunakan pendekatan ekstraksi fitur **TF-IDF** (*Term Frequency-Inverse Document Frequency*) untuk mengubah kalimat teks menjadi bobot angka sebelum melakukan klasifikasi.

1. **Multinomial Naive Bayes**
   Sebuah algoritma probabilitas berbasis *Teorema Bayes*. Model ini sangat cepat untuk dilatih dan cukup akurat untuk mengklasifikasikan dokumen berdasarkan frekuensi kemunculan kata.
   - *Akurasi Testing: ~87.8%*

2. **Logistic Regression**
   Model linear yang mencoba mencari garis pemisah terbaik antara ulasan positif dan negatif. Model ini terbukti sangat ampuh pada vektor TF-IDF karena bisa memberi bobot negatif/positif pada kata tertentu.
   - *Akurasi Testing: ~89.9%*

---

## 📊 Output Model
Setelah salah satu model dijalankan (misalnya `python naive_bayes_sentiment.py`), script tersebut akan otomatis membuat folder `output/` di dalam direktori modelnya masing-masing.

Folder `output/` tersebut akan berisi:
1. **`best_..._model.pkl`**: File model *Machine Learning* dan *TF-IDF vectorizer* yang sudah dilatih (disimpan menggunakan `joblib`). Dapat dimuat ulang di masa depan tanpa perlu melatih ulang.
2. **`evaluation_report.txt`**: Laporan evaluasi berbasis teks yang memuat nilai *Accuracy, Precision, Recall*, dan *F1-Score* untuk data latih dan data uji.
3. **`confusion_matrix.png`**: Visualisasi matriks performa yang menunjukkan jumlah prediksi benar dan salah untuk masing-masing kelas sentimen.
4. **`class_distribution.png`**: Grafik batang distribusi data awal (tersimpan terpusat di `shared_pipeline/output/`).