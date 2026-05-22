# -*- coding: utf-8 -*-
"""
=============================================================
  Data Handling & Preprocessing — Sentiment Analysis
=============================================================
"""
import io
import sys
import re
import warnings
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Set output encoding to UTF-8 for Windows compatibility
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Configuration
DATASET_DIR = Path(__file__).parent
OUTPUT_DIR = DATASET_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Predefined Stopwords & Contractions
STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","need",
    "this","that","these","those","it","its","i","me","my","we","our","you",
    "your","he","his","she","her","they","their","them","what","which","who",
    "when","where","how","all","each","both","few","more","most","other","some",
    "such","no","not","only","same","so","than","then","there","up","out","about",
    "into","through","from","by","as","if","while","although","because","since",
}

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will", "you'd": "you would",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "they're": "they are", "don't": "do not", "didn't": "did not",
}

PALETTE = {
    "Positive": "#4CAF50", "Negative": "#F44336", "Neutral": "#2196F3",
    "positive": "#4CAF50", "negative": "#F44336", "neutral": "#2196F3",
    "Irrelevant": "#FF9800", "irrelevant": "#FF9800"
}

def get_color(label: str) -> str:
    return PALETTE.get(label, "#9E9E9E")

def print_header(title: str) -> None:
    print(f"\n{'='*65}\n  {title}\n{'='*65}")

def clean_text(text: str) -> str:
    """Preprocess text: lowercase, expand contractions, remove URLs/mentions, clean non-alpha, remove stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", " ", text)  # remove HTML tags like <br />
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)

def plot_class_distribution(y: pd.Series) -> None:
    """Saves a clean class distribution plot (Class Distribution)."""
    counts = y.value_counts()
    colors = [get_color(l) for l in counts.index]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color=colors, edgecolor="black", alpha=0.8)
    plt.title("Distribusi Kelas Sentimen", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Sentimen", fontsize=12)
    plt.ylabel("Jumlah Sampel", fontsize=12)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(1, int(height*0.01)),
                 f"{height:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
                 
    plt.tight_layout()
    out = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"   -> Grafik disimpan: {out}")

def run_data_pipeline():
    """Runs simplified data preprocessing pipeline."""
    print_header("DATA PREPROCESSING & EDA")
    
    # Load dataset
    csv_file = next(DATASET_DIR.glob("*.csv"), None)
    if not csv_file:
        raise FileNotFoundError("Tidak ada file CSV di folder ini.")
    
    print(f"[OK] Membaca dataset: {csv_file.name}")
    df = pd.read_csv(csv_file, encoding="utf-8", on_bad_lines="skip")
    
    # Auto-detect text and sentiment columns
    text_col = next((c for c in ["text", "tweet", "sentence", "content", "Text", "review"] if c in df.columns), df.columns[0])
    label_col = next((c for c in ["sentiment", "label", "Sentiment", "category"] if c in df.columns), df.columns[-1])
    
    # Clean dataset
    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "sentiment"]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    
    print(f"   Ukuran dataset bersih: {len(df):,} baris")
    print(f"   Distribusi Kelas:\n{df['sentiment'].value_counts().to_string()}")
    
    # Clean text
    print("   Melakukan pembersihan teks...")
    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"].str.strip().str.len() > 0].reset_index(drop=True)
    
    # Encode label
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["sentiment"])
    classes = list(le.classes_)
    
    # Generate EDA plot (minimal: class distribution only)
    plot_class_distribution(df["sentiment"])
    
    return df, df, le, classes

if __name__ == "__main__":
    run_data_pipeline()
