# -*- coding: utf-8 -*-
"""
=============================================================
  Sentiment Analysis Dashboard — STREAMLIT DEPLOYMENT
=============================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import json
import re
import time
import sys
import traceback
from pathlib import Path

# Set page config with custom title and layout
st.set_page_config(
    page_title="Sentiment Analysis Analyzer & Comparator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS injection
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Main font override */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif;
}

/* Gradient background for stApp */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(14, 17, 23, 1) 0%, rgba(20, 24, 33, 1) 90.1%);
    color: #e2e8f0;
}

/* Header style */
.main-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-align: center;
}

.subheader-text {
    text-align: center;
    color: #94a3b8;
    font-size: 1.15rem;
    margin-bottom: 2rem;
}

/* Custom premium card design */
.premium-card {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.premium-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
}

/* Sentiment badges */
.badge {
    padding: 6px 16px;
    border-radius: 9999px;
    font-weight: 600;
    display: inline-block;
    text-align: center;
    font-size: 0.95rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.badge-positive {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.1);
}

.badge-negative {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
    box-shadow: 0 0 15px rgba(239, 68, 68, 0.1);
}

/* Status indicator dot */
.dot {
    height: 10px;
    width: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.dot-green {
    background-color: #10b981;
    box-shadow: 0 0 8px #10b981;
}
.dot-red {
    background-color: #ef4444;
    box-shadow: 0 0 8px #ef4444;
}

/* Custom table styling */
.model-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
}
.model-table th {
    background-color: rgba(79, 70, 229, 0.15);
    color: #e2e8f0;
    text-align: left;
    padding: 12px;
    border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    font-weight: 600;
}
.model-table td {
    padding: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.model-table tr:hover {
    background-color: rgba(255, 255, 255, 0.02);
}

/* Gradient buttons style overrides */
div.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    transition: all 0.2s;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
    transform: scale(1.02);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Define file paths dynamically
BASE_DIR = Path(__file__).parent.resolve()
NB_MODEL_PATH = BASE_DIR / "naive_bayes_classifier-main" / "output" / "best_naive_bayes_model.pkl"
LR_MODEL_PATH = BASE_DIR / "logistic_regression_classifier" / "output" / "best_logistic_regression_model.pkl"
LSTM_MODEL_PATH = BASE_DIR / "lstm_classifier" / "output" / "lstm_sentiment_model.pt"
LSTM_VOCAB_PATH = BASE_DIR / "lstm_classifier" / "output" / "vocabulary.json"
BERT_MODEL_DIR = BASE_DIR / "bert_classifier" / "output" / "distilbert_saved_model"

# Predefined Stopwords & Contractions (shared_pipeline)
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

def clean_text(text: str) -> str:
    """Preprocess text exactly matching data_preprocessing.py"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)

# ============================================================
#  LSTM Model Architecture & Classes
# ============================================================
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

    def encode(self, text: str, max_len: int = 150) -> list[int]:
        tokens = text.split()[:max_len]
        return [self.word2idx.get(t, 1) for t in tokens]  # 1 is <UNK>

class ManualLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.gates_x = nn.Linear(input_size, 4 * hidden_size)
        self.gates_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x, h_prev, c_prev):
        gates = self.gates_x(x) + self.gates_h(h_prev)
        i_gate = torch.sigmoid(gates[:, :self.hidden_size])
        f_gate = torch.sigmoid(gates[:, self.hidden_size:2*self.hidden_size])
        g_gate = torch.tanh(gates[:, 2*self.hidden_size:3*self.hidden_size])
        o_gate = torch.sigmoid(gates[:, 3*self.hidden_size:])
        c_new = f_gate * c_prev + i_gate * g_gate
        h_new = o_gate * torch.tanh(c_new)
        return h_new, c_new

class ManualLSTMLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = ManualLSTMCell(input_size, hidden_size)

    def forward(self, x, reverse: bool = False):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []
        indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        for t in indices:
            h, c = self.cell(x[:, t, :], h, c)
            outputs.append(h)
        if reverse:
            outputs = outputs[::-1]
        return torch.stack(outputs, dim=1), h

class ManualBiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fwd_layers = nn.ModuleList()
        self.bwd_layers = nn.ModuleList()
        self.drop_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size * 2
            self.fwd_layers.append(ManualLSTMLayer(layer_input, hidden_size))
            self.bwd_layers.append(ManualLSTMLayer(layer_input, hidden_size))
            if i < num_layers - 1 and dropout > 0:
                self.drop_layers.append(nn.Dropout(dropout))
            else:
                self.drop_layers.append(nn.Identity())

    def forward(self, x):
        for i in range(self.num_layers):
            fwd_out, fwd_h = self.fwd_layers[i](x, reverse=False)
            bwd_out, bwd_h = self.bwd_layers[i](x, reverse=True)
            x = torch.cat([fwd_out, bwd_out], dim=2)
            x = self.drop_layers[i](x)
        return x, fwd_h, bwd_h

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.bilstm = ManualBiLSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, fwd_h, bwd_h = self.bilstm(embedded)
        hidden_cat = torch.cat([fwd_h, bwd_h], dim=1)
        out = self.dropout(hidden_cat)
        out = self.fc(out)
        return out


# ============================================================
#  Model Loading Caches
# ============================================================
@st.cache_resource
def load_naive_bayes_model():
    try:
        data = joblib.load(NB_MODEL_PATH)
        return data["pipeline"], data.get("classes", ["negative", "positive"]), None
    except Exception:
        return None, None, traceback.format_exc()

@st.cache_resource
def load_logistic_regression_model():
    try:
        data = joblib.load(LR_MODEL_PATH)
        return data["pipeline"], data.get("classes", ["negative", "positive"]), None
    except Exception:
        return None, None, traceback.format_exc()

@st.cache_resource
def load_lstm_model():
    try:
        with open(LSTM_VOCAB_PATH, "r", encoding="utf-8") as f:
            word2idx = json.load(f)
        vocab = Vocabulary()
        vocab.word2idx = word2idx
        vocab.idx2word = {v: k for k, v in word2idx.items()}
        
        checkpoint = torch.load(LSTM_MODEL_PATH, map_location="cpu", weights_only=False)
        model = LSTMClassifier(
            vocab_size=checkpoint["vocab_size"],
            embed_dim=checkpoint["embed_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            num_classes=checkpoint["num_classes"],
            dropout=checkpoint["dropout"],
            pad_idx=checkpoint["pad_idx"]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, vocab, None
    except Exception:
        return None, None, traceback.format_exc()

@st.cache_resource
def load_distilbert_model():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import urllib.request
        import zipfile
        
        # Check if local model weights or config are missing
        has_weights = (BERT_MODEL_DIR / "model.safetensors").exists() or (BERT_MODEL_DIR / "pytorch_model.bin").exists()
        has_config = (BERT_MODEL_DIR / "config.json").exists()
        
        # Download from GitHub Release if missing
        if not (has_config and has_weights):
            zip_path = BASE_DIR / "bert_classifier" / "output" / "distilbert_saved_model.zip"
            url = "https://github.com/LionelSaputra/AOL-NLP/releases/download/v1.0.0/distilbert_saved_model.zip"
            try:
                BERT_MODEL_DIR.parent.mkdir(exist_ok=True, parents=True)
                urllib.request.urlretrieve(url, zip_path)
                BERT_MODEL_DIR.mkdir(exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(BERT_MODEL_DIR)
                if zip_path.exists():
                    zip_path.unlink()
                # Re-evaluate file checks after extraction
                has_weights = (BERT_MODEL_DIR / "model.safetensors").exists() or (BERT_MODEL_DIR / "pytorch_model.bin").exists()
                has_config = (BERT_MODEL_DIR / "config.json").exists()
            except Exception:
                if zip_path.exists():
                    zip_path.unlink()
                    
        # Use local if fully available, otherwise fall back to Hugging Face Hub public model
        if has_config and has_weights:
            model_path = str(BERT_MODEL_DIR)
        else:
            model_path = "lvwerra/distilbert-imdb"
            
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return model, tokenizer, None
    except Exception:
        return None, None, traceback.format_exc()


# Load all models once
nb_pipeline, nb_classes, nb_err = load_naive_bayes_model()
lr_pipeline, lr_classes, lr_err = load_logistic_regression_model()
lstm_model, lstm_vocab, lstm_err = load_lstm_model()
bert_model, bert_tokenizer, bert_err = load_distilbert_model()

# Base classes mapping
classes_mapping = lr_classes if lr_classes is not None else (nb_classes if nb_classes is not None else ["negative", "positive"])


# ============================================================
#  Prediction Helper Functions
# ============================================================
def predict_naive_bayes(text: str):
    if nb_pipeline is None:
        raise ValueError("Model Naive Bayes tidak termuat.")
    cleaned = clean_text(text)
    probs = nb_pipeline.predict_proba([cleaned])[0]
    pred_enc = np.argmax(probs)
    return nb_classes[pred_enc], probs[pred_enc]

def predict_logistic_regression(text: str):
    if lr_pipeline is None:
        raise ValueError("Model Logistic Regression tidak termuat.")
    cleaned = clean_text(text)
    probs = lr_pipeline.predict_proba([cleaned])[0]
    pred_enc = np.argmax(probs)
    return lr_classes[pred_enc], probs[pred_enc]

def predict_lstm(text: str):
    if lstm_model is None:
        raise ValueError("Model LSTM tidak termuat.")
    cleaned = clean_text(text)
    encoded = lstm_vocab.encode(cleaned, max_len=150)
    
    # Pad or truncate
    if len(encoded) < 150:
        encoded = encoded + [0] * (150 - len(encoded))
    else:
        encoded = encoded[:150]
        
    tensor_in = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        outputs = lstm_model(tensor_in)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred_enc = np.argmax(probs)
        return classes_mapping[pred_enc], probs[pred_enc]

def predict_distilbert(text: str):
    if bert_model is None or bert_tokenizer is None:
        raise ValueError("Model DistilBERT tidak termuat.")
    # DistilBERT works on raw text
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).numpy()[0]
        pred_enc = np.argmax(probs)
        confidence = probs[pred_enc]
        # Dynamic label mapping from config if available
        if hasattr(bert_model, "config") and hasattr(bert_model.config, "id2label") and bert_model.config.id2label:
            pred_label = bert_model.config.id2label[pred_enc].lower()
        else:
            pred_label = classes_mapping[pred_enc]
            
        # Standardize labels to "negative" or "positive"
        pred_label_lower = pred_label.lower()
        if "label_0" in pred_label_lower or "neg" in pred_label_lower or pred_label_lower == "0":
            pred_label = "negative"
        elif "label_1" in pred_label_lower or "pos" in pred_label_lower or pred_label_lower == "1":
            pred_label = "positive"
            
        # Fallback safeguard
        if pred_label.lower() not in ["positive", "negative"]:
            pred_label = classes_mapping[pred_enc]
            
        return pred_label, confidence


# ============================================================
#  UI Layout
# ============================================================

# Header Title
st.markdown('<div class="main-header">Sentiment Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-text">Klasifikasi Sentimen Ulasan Film IMDB dengan Perbandingan Model ML & DL</div>', unsafe_allow_html=True)

# Sidebar with model loading status
st.sidebar.markdown("### Status Pemuatan Model")
def get_status_html(name, model_obj, err):
    if model_obj is not None:
        return f'<div><span class="dot dot-green"></span><b>{name}</b>: Berhasil Dimuat</div>'
    else:
        tooltip = f'title="{err}"' if err else ""
        return f'<div {tooltip}><span class="dot dot-red"></span><b>{name}</b>: Gagal/Tidak Ditemukan</div>'

st.sidebar.markdown(get_status_html("Naive Bayes", nb_pipeline, nb_err), unsafe_allow_html=True)
st.sidebar.markdown(get_status_html("Logistic Regression", lr_pipeline, lr_err), unsafe_allow_html=True)
st.sidebar.markdown(get_status_html("Bidirectional LSTM", lstm_model, lstm_err), unsafe_allow_html=True)
st.sidebar.markdown(get_status_html("DistilBERT", bert_model, bert_err), unsafe_allow_html=True)

# Diagnostics & traceback display if any error occurs
if nb_err or lr_err or lstm_err or bert_err:
    with st.sidebar.expander("🛠️ Diagnostics & Errors", expanded=True):
        st.markdown(f"**Python:** `{sys.version.split()[0]}`")
        st.markdown(f"**PyTorch:** `{torch.__version__}`")
        st.markdown(f"**Working Dir:** `{Path.cwd().name}`")
        st.markdown("**File Existence Checks:**")
        st.markdown(f"- NB model: `{'✅' if NB_MODEL_PATH.exists() else '❌'}`")
        st.markdown(f"- LR model: `{'✅' if LR_MODEL_PATH.exists() else '❌'}`")
        st.markdown(f"- LSTM model: `{'✅' if LSTM_MODEL_PATH.exists() else '❌'}`")
        st.markdown(f"- LSTM vocab: `{'✅' if LSTM_VOCAB_PATH.exists() else '❌'}`")
        st.markdown(f"- BERT dir: `{'✅' if BERT_MODEL_DIR.exists() else '❌'}`")
        st.markdown(f"- BERT config: `{'✅' if (BERT_MODEL_DIR / 'config.json').exists() else '❌'}`")
        has_bert_weights = (BERT_MODEL_DIR / "model.safetensors").exists() or (BERT_MODEL_DIR / "pytorch_model.bin").exists()
        st.markdown(f"- BERT weights: `{'✅' if has_bert_weights else '❌'}`")
        
        if lstm_err:
            st.error("LSTM Traceback:")
            st.code(lstm_err, language="python")
        if bert_err:
            st.error("DistilBERT Traceback:")
            st.code(bert_err, language="python")
        if nb_err:
            st.error("NB Traceback:")
            st.code(nb_err, language="python")
        if lr_err:
            st.error("LR Traceback:")
            st.code(lr_err, language="python")

st.sidebar.markdown("---")
st.sidebar.markdown("### Keterangan Kelas")
st.sidebar.markdown('<span class="badge badge-positive">Positive</span>: Ulasan Menyenangkan/Bagus', unsafe_allow_html=True)
st.sidebar.markdown('<span class="badge badge-negative">Negative</span>: Ulasan Buruk/Kecewa', unsafe_allow_html=True)

# Main Application Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction & Comparison", "📊 Batch Processing (CSV)", "📖 Tentang Model"])

# ------------------------------------------------------------
#  Tab 1: Single Prediction & Comparison
# ------------------------------------------------------------
with tab1:
    st.markdown("### Uji Coba Input Teks")
    default_text = "I absolutely loved this movie! The acting was brilliant, the plot kept me on the edge of my seat, and the cinematography was stunning. Highly recommended."
    user_input = st.text_area("Masukkan teks ulasan film (Bahasa Inggris):", value=default_text, height=120)
    
    col_sel, col_action = st.columns([3, 1])
    with col_sel:
        model_selection = st.selectbox(
            "Pilih Model Analisis Sentimen:",
            ["Bandingkan Semua Model", "Naive Bayes", "Logistic Regression", "Bidirectional LSTM", "DistilBERT"]
        )
    with col_action:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("Mulai Analisis Sentimen", use_container_width=True)

    if analyze_btn or (user_input and model_selection):
        if not user_input.strip():
            st.warning("Silakan masukkan teks terlebih dahulu.")
        else:
            st.markdown("### Hasil Analisis")
            
            # Comparison view
            if model_selection == "Bandingkan Semua Model":
                cols = st.columns(4)
                models_to_test = [
                    ("Naive Bayes", predict_naive_bayes, nb_pipeline, cols[0]),
                    ("Logistic Regression", predict_logistic_regression, lr_pipeline, cols[1]),
                    ("Bidirectional LSTM", predict_lstm, lstm_model, cols[2]),
                    ("DistilBERT", predict_distilbert, bert_model, cols[3])
                ]
                
                for name, predict_fn, model_loaded, column in models_to_test:
                    with column:
                        st.markdown(f'<div class="premium-card">', unsafe_allow_html=True)
                        st.markdown(f"#### {name}")
                        if model_loaded is None:
                            st.error("Model tidak tersedia")
                        else:
                            try:
                                t_start = time.perf_counter()
                                pred, confidence = predict_fn(user_input)
                                t_end = time.perf_counter()
                                latency = (t_end - t_start) * 1000  # ms
                                
                                badge_class = "badge-positive" if pred.lower() == "positive" else "badge-negative"
                                st.markdown(f'<span class="badge {badge_class}">{pred}</span>', unsafe_allow_html=True)
                                st.markdown(f"**Confidence:** `{confidence:.2%}`")
                                st.markdown(f"**Waktu Inferensi:** `{latency:.2f} ms`")
                                
                                # Progress bar confidence
                                st.progress(float(confidence))
                            except Exception as e:
                                st.error(f"Gagal prediksi: {e}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
            # Single model view
            else:
                mapping = {
                    "Naive Bayes": (predict_naive_bayes, nb_pipeline),
                    "Logistic Regression": (predict_logistic_regression, lr_pipeline),
                    "Bidirectional LSTM": (predict_lstm, lstm_model),
                    "DistilBERT": (predict_distilbert, bert_model)
                }
                
                predict_fn, model_loaded = mapping[model_selection]
                
                if model_loaded is None:
                    st.error(f"Model {model_selection} tidak dapat dijalankan karena file model belum dimuat.")
                else:
                    try:
                        t_start = time.perf_counter()
                        pred, confidence = predict_fn(user_input)
                        t_end = time.perf_counter()
                        latency = (t_end - t_start) * 1000
                        
                        col_res, col_metric = st.columns([2, 2])
                        with col_res:
                            st.markdown(f'<div class="premium-card">', unsafe_allow_html=True)
                            st.markdown(f"#### Model: {model_selection}")
                            badge_class = "badge-positive" if pred.lower() == "positive" else "badge-negative"
                            st.markdown(f'<h3>Sentimen: <span class="badge {badge_class}">{pred}</span></h3>', unsafe_allow_html=True)
                            st.markdown(f"<h4>Tingkat Kepercayaan: <b>{confidence:.2%}</b></h4>", unsafe_allow_html=True)
                            st.progress(float(confidence))
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        with col_metric:
                            st.markdown(f'<div class="premium-card">', unsafe_allow_html=True)
                            st.markdown("#### Informasi Inferensi & Preprocessing")
                            st.markdown(f"- **Waktu inferensi**: `{latency:.2f} ms`")
                            if model_selection != "DistilBERT":
                                st.markdown("- **Preprocessing**: Teks dibersihkan (Lowercasing, Contraction Expansion, HTML & Regex Cleaning, Stopword Removal).")
                                cleaned_sample = clean_text(user_input)
                                st.text_area("Teks Bersih (Hasil Preprocessing):", value=cleaned_sample, disabled=True, height=80)
                            else:
                                st.markdown("- **Preprocessing**: Teks mentah asli dikirim langsung ke tokenizer DistilBERT (Transformer mempertahankan huruf besar/kecil dan tanda baca untuk konteks).")
                            st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")

# ------------------------------------------------------------
#  Tab 2: Batch Processing (CSV)
# ------------------------------------------------------------
with tab2:
    st.markdown("### Analisis Sentimen Skala Besar (Batch)")
    st.write("Unggah file CSV berisi ulasan film untuk dianalisis secara massal menggunakan salah satu model.")
    
    uploaded_file = st.file_uploader("Pilih file CSV (Harus memiliki kolom teks ulasan):", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("💡 **Pratinjau File yang Diunggah:**")
        st.dataframe(df.head(5), use_container_width=True)
        
        # Select column containing reviews
        cols = list(df.columns)
        selected_text_col = st.selectbox("Pilih kolom yang berisi ulasan film:", cols, index=0)
        
        batch_model = st.selectbox("Pilih Model untuk Analisis Batch:", ["Naive Bayes", "Logistic Regression", "Bidirectional LSTM", "DistilBERT"], key="batch_model")
        
        # Verify model availability
        models_available = {
            "Naive Bayes": nb_pipeline,
            "Logistic Regression": lr_pipeline,
            "Bidirectional LSTM": lstm_model,
            "DistilBERT": bert_model
        }
        
        if st.button("Mulai Klasifikasi Batch"):
            if models_available[batch_model] is None:
                st.error(f"Gagal: Model {batch_model} tidak termuat.")
            else:
                try:
                    predict_functions = {
                        "Naive Bayes": predict_naive_bayes,
                        "Logistic Regression": predict_logistic_regression,
                        "Bidirectional LSTM": predict_lstm,
                        "DistilBERT": predict_distilbert
                    }
                    pred_fn = predict_functions[batch_model]
                    
                    st.info(f"Memproses {len(df)} ulasan menggunakan {batch_model}...")
                    
                    progress_bar = st.progress(0)
                    predictions = []
                    confidences = []
                    
                    t_start = time.time()
                    for idx, row in df.iterrows():
                        text_val = str(row[selected_text_col])
                        try:
                            p_label, p_conf = pred_fn(text_val)
                        except Exception:
                            p_label, p_conf = "Error", 0.0
                        predictions.append(p_label)
                        confidences.append(p_conf)
                        # Update progress
                        progress_bar.progress((idx + 1) / len(df))
                        
                    t_total = time.time() - t_start
                    
                    # Add results to dataframe
                    df["predicted_sentiment"] = predictions
                    df["confidence"] = confidences
                    
                    st.success(f"Selesai! Klasifikasi {len(df)} data memakan waktu {t_total:.2f} detik.")
                    
                    # Layout results
                    c_left, c_right = st.columns([1, 1])
                    with c_left:
                        st.dataframe(df.head(10), use_container_width=True)
                        # CSV download button
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Unduh Hasil Sentimen (CSV)",
                            data=csv_data,
                            file_name="imdb_sentiment_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                    with c_right:
                        st.write("📊 **Distribusi Hasil Sentimen:**")
                        counts = df["predicted_sentiment"].value_counts()
                        st.bar_chart(counts)
                        
                        # Show percentage statistics
                        total_valid = len(df[df["predicted_sentiment"] != "Error"])
                        if total_valid > 0:
                            pos_count = len(df[df["predicted_sentiment"] == "positive"])
                            neg_count = len(df[df["predicted_sentiment"] == "negative"])
                            st.write(f"- **Positif (Positive)**: `{pos_count} ({pos_count/total_valid:.1%})`")
                            st.write(f"- **Negatif (Negative)**: `{neg_count} ({neg_count/total_valid:.1%})`")
                            
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses batch: {e}")

# ------------------------------------------------------------
#  Tab 3: Tentang Model
# ------------------------------------------------------------
with tab3:
    st.markdown("### Perbandingan Metrik & Detail Arsitektur")
    st.write("Berikut ringkasan hasil evaluasi dan cara kerja model berdasarkan pengerjaan proyek:")
    
    # HTML Table comparing metrics
    st.markdown("""
    <table class="model-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>Akurasi Train</th>
                <th>Akurasi Test</th>
                <th>F1-Score</th>
                <th>Waktu Pelatihan</th>
                <th>Kelebihan / Karakteristik</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><b>Naive Bayes (Multinomial)</b></td>
                <td>90.19%</td>
                <td>87.87%</td>
                <td>0.88</td>
                <td>Sangat cepat (&lt; 1 menit)</td>
                <td>Sangat ringan secara memori, baik untuk baseline awal.</td>
            </tr>
            <tr>
                <td><b>Logistic Regression</b></td>
                <td>93.53%</td>
                <td>89.94%</td>
                <td>0.90</td>
                <td>Sangat cepat (&lt; 1 menit)</td>
                <td>Sangat efisien dalam sparse matrix (TF-IDF), interpretasi baik.</td>
            </tr>
            <tr>
                <td><b>Bidirectional LSTM</b></td>
                <td>96.82%</td>
                <td>87.64%</td>
                <td>0.88</td>
                <td>225.04 menit (7 Epoch)</td>
                <td>Memahami konteks urutan kata secara bidirectional.</td>
            </tr>
            <tr>
                <td><b>DistilBERT (Transformer)</b></td>
                <td>~98.50%</td>
                <td>92.84%</td>
                <td>0.93</td>
                <td>263.38 menit (3 Epoch)</td>
                <td>Pendekatan State-of-the-Art, transfer learning yang sangat cerdas.</td>
            </tr>
        </tbody>
    </table>
    <br>
    """, unsafe_allow_html=True)
    
    with st.expander("1. Detail Model Naive Bayes"):
        st.markdown("""
        - **Feature Extraction**: TF-IDF Vectorizer (Max features: 25.000, Unigram & Bigram).
        - **Algoritma**: Probabilistik berdasar Teorema Bayes dengan asumsi independensi fitur yang kuat (naif).
        - **Kesimpulan**: Stabil, tanpa overfitting parah.
        """)
        
    with st.expander("2. Detail Model Logistic Regression"):
        st.markdown("""
        - **Feature Extraction**: TF-IDF Vectorizer (Max features: 25.000, Unigram & Bigram).
        - **Algoritma**: Klasifikasi linier yang menggunakan fungsi aktivasi Sigmoid untuk mengubah kombinasi linier input menjadi probabilitas (rentang 0-1).
        - **Kesimpulan**: Performa sangat mengejutkan mendekati 90%, mengalahkan model DL sederhana (LSTM) di dataset menengah.
        """)
        
    with st.expander("3. Detail Model Bidirectional LSTM"):
        st.markdown("""
        - **Feature Extraction**: Tokenisasi berdasarkan urutan posisi kata dalam kamus (Vocabulary Size: 25.000), diubah ke layer Embedding berdimensi 128.
        - **Algoritma**: Recurrent Neural Network dengan sel LSTM khusus (Manual cell, kompatibel dengan GPU AMD DirectML) yang memproses urutan teks dari kiri-ke-kanan dan kanan-ke-kiri.
        - **Kesimpulan**: Bagus untuk menangkap konteks kronologis kalimat, namun memerlukan waktu latih yang lama dan performanya di bawah regresi logistik pada jumlah data terbatas.
        """)
        
    with st.expander("4. Detail Model DistilBERT"):
        st.markdown("""
        - **Feature Extraction**: WordPiece Tokenizer bawaan HuggingFace. Tidak memerlukan cleaning teks agresif (mempertahankan tanda baca dan huruf kapital untuk menangkap semantik).
        - **Algoritma**: Transformer bertipe Encoder (DistilBERT) hasil penyulingan parameter BERT base (mengurangi parameter sebesar 40% namun mempertahankan 97% kecerdasan). Menggunakan *Self-Attention* global.
        - **Kesimpulan**: Model terbaik dengan akurasi 92.84%. Paling handal memahami sarkasme, idiom, dan makna tersirat.
        """)

    with st.expander("5. Deskripsi Dataset & Data Handling"):
        st.markdown("""
        - **Dataset**: Menggunakan dataset **IMDB Movie Reviews** sebanyak **50.000 ulasan film** (terbagi seimbang menjadi 25.000 ulasan positif dan 25.000 ulasan negatif). [Sumber Dataset di Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
        - **Data Handling**:
          - **Pembersihan Struktur**: Menghapus nilai kosong (`dropna`) dan baris ulasan duplikat (`drop_duplicates`) untuk mencegah bias. Label sentimen dikodekan secara biner (`0` untuk negatif, `1` untuk positif) menggunakan LabelEncoder.
          - **Pembersihan Konten (Text Cleaning)**: Mengubah huruf menjadi kecil (*lowercasing*), ekspansi singkatan (*contraction expansion*), pembersihan regex (link URL, tag HTML, mention, karakter non-alfabet), dan pembuangan kata hubung (*stopword removal*).
          - **Pengecualian**: Model **DistilBERT** sengaja melompati pembersihan konten teks ini dan langsung menggunakan teks mentah agar informasi tanda baca dan tata bahasa (grammar) tidak hilang, yang sangat penting untuk mekanisme self-attention.
        """)

    with st.expander("6. Daftar Pustaka / Referensi"):
        st.markdown("""
        - **Multinomial Naive Bayes**: McCallum, A., & Nigam, K. (1998). *A comparison of event models for Naive Bayes text classification*. AAAI-98 workshop on learning for text categorization.
        - **Logistic Regression**: Hosmer Jr, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied logistic regression*. John Wiley & Sons.
        - **LSTM**: Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory*. Neural computation, 9(8), 1735-1780.
        - **DistilBERT**: Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv preprint arXiv:1910.01108.
        - **Scikit-Learn**: Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python*. Journal of machine learning research, 12, 2825-2830.
        """)
