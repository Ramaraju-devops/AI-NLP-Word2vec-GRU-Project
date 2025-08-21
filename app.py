import os
import re
from string import digits
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer

import tensorflow as tf
#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout

# -----------------------------
# Global hyperparameters
# -----------------------------
MAX_LEN = 200
VOCAB_SIZE = 50000
EMBEDDING_DIM = 300
EMBEDDING_TRAINABLE = False
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Paths
DATASET_CSV_PRIMARY = os.path.join("./datasets", "IMDB-Dataset.csv")
# DATASET_CSV_FALLBACK = os.path.join("./datasets", "IMDB-Dataset-backup.csv")
W2V_BIN = os.path.join("./datasets/word2vec", "GoogleNews-vectors-negative300.bin")

# -----------------------------
# Utilities
# -----------------------------

def resolve_dataset_path() -> str:
    if os.path.exists(DATASET_CSV_PRIMARY):
        return DATASET_CSV_PRIMARY
    if os.path.exists(DATASET_CSV_FALLBACK):
        return DATASET_CSV_FALLBACK
    raise FileNotFoundError(
        f"Could not find dataset. Expected at '{DATASET_CSV_PRIMARY}' or '{DATASET_CSV_FALLBACK}'."
    )


def read_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    expected_cols = {"review", "sentiment"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Dataset must contain columns {expected_cols}, found: {df.columns.tolist()}"
        )
    return df


def clean_text_series(text_series: pd.Series) -> pd.Series:
    text_series = text_series.astype(str).str.lower()
    text_series = text_series.apply(lambda x: re.sub(r"([?.!,¿])", r" \1 ", x))
    text_series = text_series.apply(lambda x: re.sub(r"[\"\']", "", x))
    text_series = text_series.apply(lambda x: re.sub(r"[^a-zA-Z?.!,¿]+", " ", x))
    rm_digits = str.maketrans('', '', digits)
    text_series = text_series.apply(lambda x: x.translate(rm_digits))
    text_series = text_series.str.strip()
    text_series = text_series.apply(lambda x: re.sub(r"\s+", " ", x))
    return text_series


def build_model(vocab_size: int, embedding_dim: int, embedding_matrix: np.ndarray) -> tf.keras.Model:
    model = Sequential(name="w2v_gru_sentiment")
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=MAX_LEN,
            weights=[embedding_matrix],
            trainable=EMBEDDING_TRAINABLE,
            name="pretrained_embedding",
        )
    )
    model.add(GRU(128, name="gru"))
    model.add(Dense(128, activation="relu", name="dense_hidden"))
    model.add(Dropout(0.3, name="dropout"))
    model.add(Dense(1, activation="sigmoid", name="output"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


@st.cache_resource(show_spinner=False)
def load_w2v(bin_path: str) -> KeyedVectors:
    if not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"Word2Vec binary not found at: {bin_path}. Place GoogleNews-vectors-negative300.bin under ./datasets/word2vec/."
        )
    return KeyedVectors.load_word2vec_format(bin_path, binary=True)


@st.cache_resource(show_spinner=False)
def fit_tokenizer(texts: pd.Series, vocab_size: int) -> Tokenizer:
    tok = Tokenizer(num_words=vocab_size, lower=True, oov_token="<OOV>")
    tok.fit_on_texts(texts.tolist())
    return tok


def build_embedding_matrix(tokenizer: Tokenizer, w2v: KeyedVectors, vocab_cap: int) -> Tuple[np.ndarray, int, int]:
    word_index = tokenizer.word_index
    vocab_size_effective = min(vocab_cap, len(word_index) + 1)
    embedding_matrix = np.zeros((vocab_size_effective, EMBEDDING_DIM), dtype=np.float32)
    not_found = 0
    for word, idx in word_index.items():
        if idx >= vocab_size_effective:
            continue
        if word in w2v.key_to_index:
            embedding_matrix[idx] = w2v[word]
        else:
            not_found += 1
    return embedding_matrix, vocab_size_effective, not_found


def prepare_features(tokenizer: Tokenizer, texts: pd.Series) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts.tolist())
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    return X


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="IMDb Sentiment (W2V + GRU)", layout="wide")

st.title("IMDb Sentiment Analysis - Word2Vec + GRU")

with st.sidebar:
    st.header("Configuration")
    epochs = st.number_input("Epochs", min_value=1, max_value=10, value=2, step=1)
    batch_size = st.number_input("Batch size", min_value=16, max_value=256, value=64, step=16)
    sample_limit = st.number_input(
        "Training sample limit (0 = full dataset)", min_value=0, max_value=50000, value=2000, step=500
    )
    st.caption("Tip: use a small subset first to verify the setup, then train on full data.")

# Load & clean dataset
try:
    dataset_path = resolve_dataset_path()
    df = read_data(dataset_path)
    df["review"] = clean_text_series(df["review"])
    # Map labels to 0/1 if they are strings
    if df["sentiment"].dtype == object:
        df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1}).astype(int)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Optional subset for faster runs
if sample_limit and sample_limit > 0:
    df_use = df.sample(n=min(sample_limit, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
else:
    df_use = df

# Tokenizer
tokenizer = fit_tokenizer(df_use["review"], VOCAB_SIZE)
X_all = prepare_features(tokenizer, df_use["review"])  # features for train/test split
y_all = df_use["sentiment"].to_numpy().astype(np.int32)

# Load W2V and build embedding matrix
with st.spinner("Loading Word2Vec (first time may take a minute)..."):
    w2v = load_w2v(W2V_BIN)
embedding_matrix, vocab_size_eff, not_found = build_embedding_matrix(tokenizer, w2v, VOCAB_SIZE)

col1, col2, col3 = st.columns(3)
col1.metric("Samples", f"{len(df_use):,}")
col2.metric("Vocab (effective)", f"{vocab_size_eff:,}")
col3.metric("OOV (within cap)", f"{not_found:,}")

# Train button
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = tokenizer

train_clicked = st.button("Train / Re-train Model", type="primary")

if train_clicked:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
    )

    model = build_model(vocab_size_eff, EMBEDDING_DIM, embedding_matrix)

    with st.spinner("Training model..."):
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

    with st.spinner("Evaluating..."):
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred_prob = model.predict(X_test, batch_size=batch_size, verbose=0)
        y_pred_label = (y_pred_prob >= 0.5).astype(int).flatten()
        manual_acc = accuracy_score(y_test, y_pred_label)
        report = classification_report(y_test, y_pred_label, target_names=["negative", "positive"])

    st.success(f"Eval accuracy: {acc*100:.2f}% | Loss: {loss:.4f} (manual acc: {manual_acc*100:.2f}%)")
    with st.expander("Classification report"):
        st.text(report)

    st.session_state.model = model
    st.session_state.tokenizer = tokenizer

st.subheader("Try a review")
user_text = st.text_area("Enter a movie review:", height=150, placeholder="Type a review and click Predict...")

colp1, colp2 = st.columns([1, 3])
if colp1.button("Predict"):
    if st.session_state.model is None:
        st.warning("Train the model first using the button above.")
    else:
        user_clean = clean_text_series(pd.Series([user_text]))
        seq = st.session_state.tokenizer.texts_to_sequences(user_clean.tolist())
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        prob = float(st.session_state.model.predict(pad, verbose=0)[0][0])
        label = "positive" if prob >= 0.5 else "negative"
        colp2.success(f"Prediction: {label} (Prob(positive)={prob:.3f})")

st.caption(
    "Run locally: `streamlit run app.py`. Ensure the dataset and GoogleNews Word2Vec binary exist under the ./datasets folder."
)
