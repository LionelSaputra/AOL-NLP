# -*- coding: utf-8 -*-
"""
=============================================================
  Naive Bayes Sentiment Analysis — MODEL
=============================================================
"""
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
