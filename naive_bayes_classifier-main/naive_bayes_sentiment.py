# ==============================================================================
#  NOTES & RUBRIC COMPLIANCE (CATATAN KEPATUHAN RUBRIK PENILAIAN)
# ==============================================================================
# 1. DESKRIPSI DATASET & SUMBER LINK:
#    - Nama: IMDB Dataset of 50K Movie Reviews
#    - Ukuran: 50.000 ulasan film (distribusi seimbang: 25.000 positif, 25.000 negatif).
#    - Sumber Link: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
# 2. JUSTIFIKASI & DESKRIPSI MODEL:
#    - Model: Multinomial Naive Bayes (MultinomialNB).
#    - Justifikasi: Dipilih karena sangat ringan secara memori, efisien, cepat dalam waktu training
#      (sangat cocok sebagai model baseline/pembanding awal dalam tugas pemrosesan bahasa alami).
# 3. DETAIL ARSITEKTUR & PENJELASAN PREPROCESSING (DATA HANDLING):
#    - Data Handling: Menggunakan shared pipeline (data_preprocessing.py) untuk dropna() (data kosong)
#      dan drop_duplicates() (data duplikat). Label sentimen diubah menjadi biner (0=negatif, 1=positif).
#    - Preprocessing Teks: Mengubah ke huruf kecil, memperluas singkatan (contraction), menghapus link URL,
#      menghapus tag HTML (<br />), membuang karakter non-alfabet, dan membuang stopwords umum.
#    - Representasi Fitur: TF-IDF Vectorizer (mengubah teks menjadi matriks bobot statistik).
# 4. KONFIGURASI HYPERPARAMETER (SKENARIO EKSPERIMEN):
#    - TF-IDF: max_features=25.000 (top-N fitur kata), ngram_range=(1, 2) (Unigram & Bigram).
#    - Naive Bayes: alpha=0.1 (Laplace smoothing parameter).
# 5. SKENARIO EVALUASI & PERBANDINGAN APPLE-TO-APPLE (TRAIN VS TEST):
#    - Pembagian Data: 80% Training (40.000 ulasan) dan 20% Testing (10.000 ulasan).
#    - Parameter Split: test_size=0.2, random_state=42 (seed yang sama untuk perbandingan adil antar model).
#    - Metrik Keluaran: Menghasilkan Akurasi Data Train (90.19%) dan Akurasi Data Test (87.87%) secara komparatif.
#    - Visualisasi: Confusion Matrix disimpan di `output/confusion_matrix.png` dan evaluasi lengkap di `output/evaluation_report.txt`.
# 6. DAFTAR PUSTAKA / REFERENSI:
#    - McCallum, A., & Nigam, K. (1998). A comparison of event models for Naive Bayes text classification.
#    - Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR.
# ==============================================================================
import io
import sys
import warnings
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Tambahkan path ke folder shared_pipeline agar bisa diimport
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "shared_pipeline"))

# Import from preprocessing module
# pyrefly: ignore [missing-import]
from data_preprocessing import (
    run_data_pipeline,
    clean_text,
    print_header,
    RANDOM_STATE,
    TEST_SIZE,
)

# Set up local output directory for this model
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

def plot_confusion_matrix(cm, labels) -> None:
    """Plots and saves exactly one Confusion Matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    out = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"   -> Grafik disimpan: {out}")

def main() -> None:
    # 1. Run Data Preprocessing Pipeline
    df, _, le, classes = run_data_pipeline()
    
    # 2. Split Data into Train/Test Sets
    print_header("TRAIN / TEST SPLIT")
    X = df["text_clean"]
    y = df["label_enc"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"   Train Size: {len(X_train):,} | Test Size: {len(X_test):,}")
    
    # 3. Build & Train Naive Bayes Model
    print_header("TRAINING NAIVE BAYES MODEL")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=25000, ngram_range=(1, 2))),
        ("clf", MultinomialNB(alpha=0.1))
    ])
    
    print("   Melatih model Multinomial Naive Bayes + TF-IDF...")
    pipeline.fit(X_train, y_train)
    print("   [OK] Training selesai.")
    
    # 4. Model Evaluation
    print_header("EVALUASI MODEL")
    y_pred = pipeline.predict(X_test)
    
    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=classes, zero_division=0)
    
    print(f"   Akurasi Data Train : {train_acc:.2%}")
    print(f"   Akurasi Data Test  : {test_acc:.2%}")
    print("\n   Laporan Klasifikasi (Test Set):")
    print(report_str)
    
    # Save metrics to text file in output folder
    report_path = OUTPUT_DIR / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=============================================================\n")
        f.write("  EVALUASI MODEL (NAIVE BAYES)\n")
        f.write("=============================================================\n\n")
        f.write(f"Akurasi Data Train : {train_acc:.2%}\n")
        f.write(f"Akurasi Data Test  : {test_acc:.2%}\n\n")
        f.write("Laporan Klasifikasi:\n")
        f.write("-" * 55 + "\n")
        f.write(report_str)
    print(f"   -> Laporan evaluasi disimpan: {report_path}")
    
    # Generate exactly 1 Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes)
    
    # 5. Demo Predictions
    print_header("DEMO PREDIKSI SENTIMEN")
    demo_sentences = [
        "I absolutely love this product, it's fantastic!",
        "This is terrible, worst experience I've ever had.",
        "The package arrived today, it's okay I guess.",
        "Great service and fast delivery, highly recommended!"
    ]
    
    print(f"   {'No.':<4} {'Teks Masukan':<55} {'Prediksi':<12} {'Confidence':>10}")
    print("   " + "-" * 85)
    for i, sent in enumerate(demo_sentences, 1):
        cleaned = clean_text(sent)
        pred_enc = pipeline.predict([cleaned])[0]
        pred_prob = pipeline.predict_proba([cleaned])[0]
        pred_label = le.inverse_transform([pred_enc])[0]
        confidence = pred_prob.max()
        print(f"   {i:<4} {sent[:53]:<55} {pred_label:<12} {confidence:>10.2%}")
        
    # 6. Save Model
    print_header("MENYIMPAN MODEL")
    try:
        # pyrefly: ignore [missing-import]
        import joblib
        model_path = OUTPUT_DIR / "best_naive_bayes_model.pkl"
        joblib.dump({
            "pipeline": pipeline,
            "label_encoder": le,
            "classes": classes,
        }, model_path)
        print(f"   [OK] Model disimpan: {model_path}")
    except Exception as e:
        print(f"   [!] Gagal menyimpan model: {e}")

if __name__ == "__main__":
    main()
