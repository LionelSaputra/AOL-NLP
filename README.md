# Sentiment Analysis NLP Pipeline

Repositori ini berisi implementasi dari saluran pipa (*pipeline*) *NLP (Natural Language Processing)* untuk menganalisis sentimen dari ulasan teks (positif/negatif) menggunakan bahasa Python. Proyek ini dibangun dengan struktur yang modular sehingga berbagai model—mulai dari *Traditional Machine Learning* hingga *Deep Learning (State-of-the-Art)*—dapat dilatih menggunakan dataset yang sama.

Dataset utama yang digunakan dalam proyek ini adalah **IMDB Movie Reviews (50.000 data)**.

---

## 📂 Struktur Proyek
Proyek ini dibagi menjadi beberapa bagian utama:
1. **`shared_pipeline/`** : Modul utama untuk memuat data (`dropna`, `drop_duplicates`) dan pembersihan teks standar.
2. **`naive_bayes_classifier-main/`** : Implementasi model *Multinomial Naive Bayes* (TF-IDF).
3. **`logistic_regression_classifier/`** : Implementasi model *Logistic Regression* (TF-IDF).
4. **`lstm_classifier/`** : Model Deep Learning berbasis *Bidirectional LSTM* yang dilatih dari awal (*from scratch*) dengan akselerasi AMD GPU (DirectML).
5. **`bert_classifier/`** : Model Deep Learning *State-of-the-Art* berbasis *DistilBERT* (Fine-Tuning Transformer) dengan akselerasi AMD GPU (DirectML).
6. **`penjelasan_model.txt`** : Dokumen penjelasan lengkap mengenai cara kerja, detail algoritma, preprocessing lanjutan, alasan pemilihan, dan output dari masing-masing model.

---

## 🧹 Data Preprocessing & Pipeline (`shared_pipeline`)
Sebelum ulasan diumpankan ke model, data mentah melewati *shared pipeline* (`data_preprocessing.py`) yang melakukan langkah-barang berikut:
* **Struktur Data**: Membaca CSV, menghapus nilai kosong (`dropna`), membuang baris teks duplikat (`drop_duplicates`), dan mengubah label sentimen teks menjadi biner (`LabelEncoder`: 0 untuk negatif, 1 untuk positif).
* **Pembersihan Konten Teks**: Mengubah teks ke huruf kecil (*lowercasing*), memperluas singkatan (*contraction expansion*), membersihkan tautan/tag HTML/angka/tanda baca (*regex cleaning*), serta membuang kata umum yang tidak sensitif terhadap sentimen (*stopword removal*).

> 💡 **Pengecualian**: Model **DistilBERT** sengaja melewatkan tahap *Pembersihan Konten Teks* dan menggunakan teks asli secara langsung untuk menjaga informasi tanda baca, kapitalisasi, dan tata bahasa guna interpretasi konteks kalimat yang optimal melalui Tokenizer-nya.

---

## 🤖 Perbandingan Performa Model

### 1. Traditional Machine Learning (Pendekatan TF-IDF)
Mengubah teks bersih menjadi vektor numerik berbasis frekuensi kata menggunakan TF-IDF (Unigram & Bigram, max 25.000 fitur).

* **Multinomial Naive Bayes**
  * Klasifikasi probabilistik sederhana berbasis Teorema Bayes dengan asumsi independensi fitur. Sangat cepat dan efisien.
  * **Akurasi Uji: 87.87%**
* **Logistic Regression**
  * Klasifikasi linier yang memberikan bobot matematis untuk setiap fitur kata yang dilewatkan ke fungsi Sigmoid.
  * **Akurasi Uji: 89.94%**

### 2. Deep Learning (Pendekatan Sekuensial & Transformer)
* **Bidirectional LSTM (Stacked)**
  * Model jaringan saraf tiruan (RNN) yang membaca teks secara sekuensial (berurutan) dari dua arah (kiri-ke-kanan & kanan-ke-kiri). Diimplementasikan secara *manual* menggunakan operasi dasar PyTorch agar kompatibel untuk diakselerasi di GPU AMD via DirectML.
  * **Akurasi Uji: 86.82%**
* **DistilBERT (Transformer)**
  * Model Transformer *pre-trained* yang memanfaatkan mekanisme *Self-Attention* global untuk memproses kalimat secara utuh sekaligus. Dilakukan *fine-tuning* pada dataset IMDB.
  * **Akurasi Uji: 92.84%** (Model terbaik)

---

## 📊 Hasil Output Model
Setiap model akan menghasilkan folder `output/` masing-masing setelah dieksekusi:

* **Untuk Model ML Tradisional (Naive Bayes & Logistic Regression)**:
  * `best_..._model.pkl`: Objek pipeline model + TF-IDF yang sudah dilatih (menggunakan `joblib`).
  * `evaluation_report.txt` & `confusion_matrix.png`: Laporan akurasi dan visualisasi matriks kesalahan prediksi.

* **Untuk LSTM**:
  * `lstm_sentiment_model.pt`: Bobot model terbaik.
  * `vocabulary.json`: Pemetaan kamus kata ke angka indeks.
  * `training_history.png`: Grafik nilai *loss* dan akurasi per epoch latihan.
  * `evaluation_report.txt` & `confusion_matrix.png`.

* **Untuk DistilBERT**:
  * `distilbert_saved_model/`: Folder berisi konfigurasi, bobot model (`model.safetensors`), dan tokenizernya yang dapat langsung di-load menggunakan HuggingFace `AutoModel`.
  * `evaluation_report.txt` & `confusion_matrix.png`.