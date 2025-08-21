# =============================================
# IMDb Sentiment Analysis with pre-trained Word2Vec + GRU (TensorFlow + sklearn)
# =============================================
#
# This script performs end-to-end sentiment analysis on the IMDb dataset using:
# - Text preprocessing (lowercasing, punctuation spacing, cleanup)
# - Label encoding (positive/negative -> 1/0)
# - Tokenization and padding
# - Pre-trained Word2Vec embeddings (GoogleNews 300d) loaded via gensim from URL
# - A GRU-based neural network with Dense and Dropout layers
# - Training, evaluation, and sample predictions on unseen text
#
# Requirements (install if missing):
#   pip install tensorflow scikit-learn gensim pandas numpy
#
# Input files expected:
#   - Dataset CSV:  ./datasets/imdb-dataset.csv
#       Columns expected: 'review' (text), 'sentiment' ("positive"/"negative")
#
# Notes:
# - Word2Vec vectors are automatically downloaded from gensim's model hub
# - If you want to fine-tune embeddings, set EMBEDDING_TRAINABLE = True.
# - Adjust MAX_LEN, VOCAB_SIZE, EPOCHS, and BATCH_SIZE as needed.
# =============================================

import os
import re
import numpy as np
import pandas as pd
from string import digits

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout

# Gensim for loading Word2Vec from URL
import gensim.downloader as api

# ... existing code ...

def list_available_w2v_models():
    """List available Word2Vec models from gensim hub"""
    print("Available Word2Vec models:")
    try:
        models = api.info()['models']
        w2v_models = [model for model in models.keys() if 'word2vec' in model.lower()]
        for model in w2v_models[:10]:  # Show first 10
            print(f"  - {model}")
        if len(w2v_models) > 10:
            print(f"  ... and {len(w2v_models) - 10} more")
    except Exception as e:
        print(f"Could not fetch model list: {e}")
        print("Common models: word2vec-google-news-300, word2vec-google-news-100")

def get_model_info(model_name):
    """Get information about a specific model"""
    try:
        info = api.info()
        if model_name in info['models']:
            model_info = info['models'][model_name]
            size_mb = model_info.get('file_size', 0) / (1024 * 1024)
            return f"{model_name} (~{size_mb:.1f}MB)"
        return model_name
    except:
        return model_name

# -----------------------------
# Global hyperparameters
# -----------------------------
DATASET_CSV = os.path.join("./datasets", "imdb-dataset.csv")

# Word2Vec model configuration
# Choose your preferred model (larger = better quality, smaller = faster download)
W2V_MODEL_NAME = 'word2vec-google-news-300'  # 300 dimensions, ~3GB, best quality
# W2V_MODEL_NAME = 'word2vec-google-news-100'  # 100 dimensions, ~100MB, good quality
# W2V_MODEL_NAME = 'word2vec-google-news-50'   # 50 dimensions, ~50MB, decent quality
# W2V_MODEL_NAME = 'glove-twitter-25'          # 25 dimensions, ~25MB, fast download

# Alternative models (smaller, faster to download) - used as fallbacks
ALTERNATIVE_MODELS = [
    'word2vec-google-news-100',  # 100 dimensions, ~100MB
    'word2vec-google-news-50',   # 50 dimensions, ~50MB
    'glove-twitter-25',          # 25 dimensions, ~25MB
]

MAX_LEN = 200             # Max tokens per review (pad/truncate)
VOCAB_SIZE = 50000        # Limit vocabulary size (most frequent words)
EMBEDDING_DIM = 300       # GoogleNews vectors are 300-dim
EMBEDDING_TRAINABLE = False
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 3
BATCH_SIZE = 64


# =============================================
# Step 1: Load dataset
# =============================================

def read_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset CSV not found at: {file_path}")
    df = pd.read_csv(file_path)
    expected_cols = {"review", "sentiment"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Dataset must contain columns {expected_cols}, found: {df.columns.tolist()}")
    return df


df = read_data(DATASET_CSV)
print("Loaded dataset shape:", df.shape)
print(df.head(2))


# =============================================
# Step 2: Preprocess text
# =============================================

def clean_text_series(text_series: pd.Series) -> pd.Series:
    # Lowercase
    text_series = text_series.astype(str).str.lower()

    # Space around punctuation and normalize spaces
    text_series = text_series.apply(lambda x: re.sub(r"([?.!,¿])", r" \1 ", x))
    text_series = text_series.apply(lambda x: re.sub(r"[\"\']", "", x))  # remove straight quotes

    # Keep letters and selected punctuation
    text_series = text_series.apply(lambda x: re.sub(r"[^a-zA-Z?.!,¿]+", " ", x))

    # Remove digits
    rm_digits = str.maketrans('', '', digits)
    text_series = text_series.apply(lambda x: x.translate(rm_digits))

    # Strip and reduce multiple spaces
    text_series = text_series.str.strip()
    text_series = text_series.apply(lambda x: re.sub(r"\s+", " ", x))
    return text_series


df["review"] = clean_text_series(df["review"])
print("Sample cleaned review:\n", df["review"].iloc[0][:300], "...")


# =============================================
# Step 3: Encode sentiment labels to 0/1
# =============================================

lb = LabelBinarizer()
df["sentiment"] = lb.fit_transform(df["sentiment"])  # positive=1, negative=0
print("Label classes:", getattr(lb, 'classes_', None))


# =============================================
# Step 4: Tokenization and Padding
# =============================================

tokenizer = Tokenizer(num_words=VOCAB_SIZE, lower=True, oov_token="<OOV>")
tokenizer.fit_on_texts(df["review"].tolist())

# Convert to sequences, then pad
sequences = tokenizer.texts_to_sequences(df["review"].tolist())
X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
y = df["sentiment"].to_numpy().astype(np.int32)

word_index = tokenizer.word_index  # dict: token -> index
vocab_size_effective = min(VOCAB_SIZE, len(word_index) + 1)  # +1 for padding idx 0
print("Vocabulary size (effective):", vocab_size_effective)
print("X shape:", X.shape, "y shape:", y.shape)


# =============================================
# Step 5: Load pre-trained Word2Vec vectors from URL and build embedding matrix
# =============================================

# Show available models
list_available_w2v_models()

print(f"\nDownloading Word2Vec model: {get_model_info(W2V_MODEL_NAME)}")
print("Note: This may take a few minutes on first run as it downloads the model data.")
print("Subsequent runs will use cached version.")

try:
    w2v = api.load(W2V_MODEL_NAME)
    print(f"✓ Word2Vec model loaded successfully. Vocab size: {len(w2v.key_to_index)}")
    print(f"✓ Embedding dimension: {EMBEDDING_DIM}")
except Exception as e:
        print(f"✗ Error loading Word2Vec model '{W2V_MODEL_NAME}': {e}")
        print("Trying alternative models...")
        
        # Try alternative models in order of preference
        for alt_model in ALTERNATIVE_MODELS:
            try:
                print(f"Attempting to load: {get_model_info(alt_model)}")
                w2v = api.load(alt_model)
                
                # Update embedding dimension based on model
                if '100' in alt_model:
                    EMBEDDING_DIM = 100
                elif '50' in alt_model:
                    EMBEDDING_DIM = 50
                elif '25' in alt_model:
                    EMBEDDING_DIM = 25
                else:
                    # Get actual dimension from model
                    EMBEDDING_DIM = w2v.vector_size
                
                print(f"✓ Loaded alternative model '{alt_model}' with {EMBEDDING_DIM} dimensions")
                print(f"✓ Vocab size: {len(w2v.key_to_index)}")
                break
                
            except Exception as e2:
                print(f"✗ Failed to load '{alt_model}': {e2}")
                continue
        else:
            print("✗ Failed to load any alternative Word2Vec model")
            print("Available alternatives:")
            list_available_w2v_models()
            raise RuntimeError("No Word2Vec model could be loaded")

# Build embedding matrix for our tokenizer vocab
embedding_matrix = np.zeros((vocab_size_effective, EMBEDDING_DIM), dtype=np.float32)
not_found = 0
for word, idx in word_index.items():
    if idx >= vocab_size_effective:
        continue
    if word in w2v.key_to_index:
        embedding_matrix[idx] = w2v[word]
    else:
        not_found += 1
print(f"Embedding matrix shape: {embedding_matrix.shape} | OOV tokens (within cap): {not_found}")


# =============================================
# Step 6: Train-test split
# =============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print("Train shapes:", X_train.shape, y_train.shape)
print("Test shapes:", X_test.shape, y_test.shape)


# =============================================
# Step 7: Build GRU model
# =============================================

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
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Build model with current embedding dimension (may have changed if fallback model was used)
model = build_model(vocab_size_effective, EMBEDDING_DIM, embedding_matrix)
model.summary()

# Print final model configuration
print(f"\n" + "="*60)
print(f"FINAL MODEL CONFIGURATION")
print(f"="*60)
print(f"Word2Vec model: {W2V_MODEL_NAME}")
print(f"Embedding dimensions: {EMBEDDING_DIM}")
print(f"Vocabulary size: {vocab_size_effective}")
print(f"Max sequence length: {MAX_LEN}")
print(f"Embeddings trainable: {EMBEDDING_TRAINABLE}")
print(f"="*60)


# =============================================
# Step 8: Train
# =============================================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
)


# =============================================
# Step 9: Evaluate
# =============================================

loss, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Accuracy (Keras evaluate): {acc*100:.2f}% | Loss: {loss:.4f}")

# Manual accuracy for sanity check
y_pred_prob = model.predict(X_test, batch_size=BATCH_SIZE)
y_pred_label = (y_pred_prob >= 0.5).astype(int)
manual_acc = accuracy_score(y_test, y_pred_label)
print(f"Test Accuracy (sklearn manual): {manual_acc*100:.2f}%")
print("\nClassification report:\n", classification_report(y_test, y_pred_label, target_names=["negative", "positive"]))


# =============================================
# Step 10: Sample predictions on unseen text
# =============================================

# sample_texts = [
#     "This movie was absolutely fantastic! The performances were stunning and the story was gripping.",
#     "Terrible. I wasted two hours of my life. The plot was dull and the acting was worse.",
#     "Not bad, but it could have been better. Some parts were enjoyable though.",
#     "A masterpiece that will be remembered for years!",
#     "I wouldn't recommend this to anyone."
# ]

# # Preprocess -> tokenize -> pad
# sample_clean = clean_text_series(pd.Series(sample_texts)).tolist()
# sample_seq = tokenizer.texts_to_sequences(sample_clean)
# sample_pad = pad_sequences(sample_seq, maxlen=MAX_LEN, padding="post", truncating="post")

# sample_probs = model.predict(sample_pad)
# sample_labels = (sample_probs >= 0.5).astype(int).flatten()

# print("\nSample predictions:")
# for txt, prob, lab in zip(sample_texts, sample_probs.flatten(), sample_labels):
#     pred = "positive" if lab == 1 else "negative"
#     print(f"- Text: {txt[:80]}...\n  Prob(positive)={prob:.3f} -> Pred={pred}")

# ---------------------------------------------
# Interactive testing: request sample input
# ---------------------------------------------
print("\nInteractive testing (type 'q' to quit).")
while True:
    try:
        user_text = input("Enter a review (or 'q' to quit): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting interactive testing.")
        break
    if not user_text or user_text.lower() in {"q", "quit", "exit"}:
        print("Exiting interactive testing.")
        break

    # Preprocess -> tokenize -> pad using same pipeline
    user_clean = clean_text_series(pd.Series([user_text])).tolist()
    user_seq = tokenizer.texts_to_sequences(user_clean)
    user_pad = pad_sequences(user_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    user_prob = float(model.predict(user_pad, verbose=0)[0][0])
    user_label = "positive" if user_prob >= 0.5 else "negative"
    print(f"Prob(positive)={user_prob:.3f} -> Pred={user_label}")
