# IMDb Sentiment Analysis with Word2Vec + GRU

This project performs sentiment analysis on the IMDb dataset using pre-trained Word2Vec embeddings and a GRU neural network. The code has been refined to automatically download Word2Vec vectors from gensim's model hub instead of requiring local files.

## Features

- **Automatic Word2Vec Download**: No need to manually download Word2Vec binary files
- **Multiple Model Options**: Choose from different sized models (25d to 300d vectors)
- **Fallback Support**: Automatically tries alternative models if the primary one fails
- **Interactive Testing**: Test the model with your own text input
- **Comprehensive Preprocessing**: Text cleaning, tokenization, and padding

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Word2Vec Loading

Before running the main script, test that Word2Vec models can be downloaded:

```bash
python test_word2vec_loading.py
```

### 3. Run the Main Script

```bash
python neural-networks-word2vec.py
```

## Configuration

### Word2Vec Model Selection

Edit the `W2V_MODEL_NAME` variable in `neural-networks-word2vec.py`:

```python
# Choose your preferred model (larger = better quality, smaller = faster download)
W2V_MODEL_NAME = 'word2vec-google-news-300'  # 300 dimensions, ~3GB, best quality
# W2V_MODEL_NAME = 'word2vec-google-news-100'  # 100 dimensions, ~100MB, good quality
# W2V_MODEL_NAME = 'word2vec-google-news-50'   # 50 dimensions, ~50MB, decent quality
# W2V_MODEL_NAME = 'glove-twitter-25'          # 25 dimensions, ~25MB, fast download
```

### Model Parameters

```python
MAX_LEN = 200             # Max tokens per review
VOCAB_SIZE = 50000        # Vocabulary size limit
EMBEDDING_DIM = 300       # Vector dimensions (auto-adjusted)
EPOCHS = 3                # Training epochs
BATCH_SIZE = 64           # Batch size
```

## How It Works

1. **Text Preprocessing**: Cleans and normalizes review text
2. **Word2Vec Loading**: Downloads pre-trained vectors from gensim hub
3. **Embedding Matrix**: Creates embedding matrix for the model vocabulary
4. **GRU Model**: Builds neural network with Embedding → GRU → Dense layers
5. **Training**: Trains on IMDb dataset with sentiment labels
6. **Evaluation**: Tests on held-out data and provides interactive testing

## Available Word2Vec Models

The script automatically shows available models and their sizes:

- `word2vec-google-news-300`: Best quality, ~3GB download
- `word2vec-google-news-100`: Good quality, ~100MB download  
- `word2vec-google-news-50`: Decent quality, ~50MB download
- `glove-twitter-25`: Fast download, ~25MB

## Dataset Requirements

The script expects a CSV file at `./datasets/imdb-dataset.csv` with columns:
- `review`: Text content of the review
- `sentiment`: "positive" or "negative"

## Error Handling

- **Automatic Fallback**: If the primary model fails, tries smaller alternatives
- **Progress Information**: Shows download progress and model information
- **Detailed Logging**: Provides clear feedback on each step

## Performance Tips

- **First Run**: Will download Word2Vec model (may take 5-15 minutes)
- **Subsequent Runs**: Uses cached models (much faster)
- **GPU Support**: Uncomment `tensorflow-gpu` in requirements.txt for faster training
- **Model Size**: Smaller models train faster but may have lower accuracy

## Troubleshooting

### Common Issues

1. **Download Failures**: Check internet connection and try smaller models
2. **Memory Issues**: Use smaller Word2Vec models (50d or 25d)
3. **Import Errors**: Ensure all dependencies are installed with correct versions

### Testing

Run the test script to verify setup:

```bash
python test_word2vec_loading.py
```

## Output

The script provides:
- Model training progress
- Test accuracy and classification report
- Interactive text input for testing
- Final model configuration summary

## License

This project is for educational purposes. The Word2Vec models are pre-trained on Google News data.
