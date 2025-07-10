# Toxic Comment Classification – My Approach

## Problem Statement (in my words)
Online platforms are full of nasty, toxic comments—stuff that’s not just rude, but can be outright hateful or dangerous. I wanted to build a system that can spot these toxic comments automatically, across multiple categories (toxic, severe_toxic, obscene, threat, insult, identity_hate), using real data from the Jigsaw Kaggle challenge.

## My Approach

### 1. Data Exploration & Preprocessing
- **Dataset:** Used the Jigsaw Toxic Comment Classification dataset (160k+ comments, multi-label).
- **Cleaning:** Lowercased, stripped URLs, HTML, special chars, and extra spaces.
- **Feature Engineering:** Added features like comment length, word count, exclamation/question marks, uppercase words, sentiment score (TextBlob), and a custom “bad words” count (see `bad_words.txt`).
- **Multi-label:** Many comments hit multiple toxic categories, so I kept that structure.

### 2. Baseline Models
- **TF-IDF + Logistic Regression:** Built a baseline using TF-IDF (unigrams & bigrams, 5k features) + engineered features, with One-vs-Rest Logistic Regression.
- **Random Forest:** Tried a Random Forest baseline for comparison.

### 3. BERT Fine-Tuning
- **Model:** Used `bert-base-uncased` as the backbone, fine-tuned for multi-label classification (6 outputs, sigmoid activation).
- **Tokenization:** Used HuggingFace’s `BertTokenizerFast`, max length 128.
- **Training:** Trained on a subset (for speed), 3 epochs, AdamW optimizer, learning rate 2e-5, with a linear scheduler.
- **Quantization:** Applied dynamic quantization to the final model for faster inference (see `bert_toxic_model_quantized.pt`).

### 4. Deployment
- **Streamlit App:** Built a simple UI (`app.py`) where you paste a comment and get toxicity probabilities for each class.
- **Explainability:** Used LIME to interpret model predictions on sample comments.

## Results

### Baseline (Logistic Regression + TF-IDF + Features)
| Label          | Precision | Recall | F1    |
|--------------- |-----------|--------|-------|
| toxic          | 0.54      | 0.86   | 0.66  |
| severe_toxic   | 0.20      | 0.90   | 0.33  |
| obscene        | 0.53      | 0.89   | 0.66  |
| threat         | 0.07      | 0.88   | 0.14  |
| insult         | 0.41      | 0.87   | 0.56  |
| identity_hate  | 0.12      | 0.82   | 0.21  |

- Micro F1: 0.55
- Macro F1: 0.43

### Random Forest Baseline
| Label          | Precision | Recall | F1    |
|--------------- |-----------|--------|-------|
| toxic          | 0.91      | 0.52   | 0.66  |
| severe_toxic   | 0.59      | 0.17   | 0.27  |
| obscene        | 0.90      | 0.56   | 0.69  |
| threat         | 0.50      | 0.05   | 0.10  |
| insult         | 0.82      | 0.43   | 0.56  |
| identity_hate  | 0.76      | 0.06   | 0.12  |

### BERT (Fine-tuned, Subset)
| Label          | Precision | Recall | F1    |
|--------------- |-----------|--------|-------|
| toxic          | 1.00      | 0.40   | 0.57  |
| severe_toxic   | 0.00      | 0.00   | 0.00  |
| obscene        | 0.00      | 0.00   | 0.00  |
| threat         | 0.00      | 0.00   | 0.00  |
| insult         | 1.00      | 0.08   | 0.15  |
| identity_hate  | 0.00      | 0.00   | 0.00  |

- Note: BERT was only trained on a small subset for demo speed, so results are not fully representative. With full data and more epochs, it should outperform the baselines.

### Qualitative
- The model is good at flagging obvious toxicity, but rare classes (like “threat”) are tough without more data or balancing.

## Key Learnings

- Data Imbalance is Real: Rare toxic classes are hard to predict. Need more data or smarter sampling.
- Feature Engineering Still Matters: Even with BERT, engineered features (like bad word count) help in classic models.
- BERT is Powerful, but Resource-Hungry: Fine-tuning on full data is slow and needs a good GPU.
- Quantization: Easy win for faster inference, especially for deployment.
- Explainability: LIME is handy for debugging and trust, but can be slow on big models.

## Project Structure

```
toxic_comment_classification/
│
├── app.py                        # Streamlit app for live demo
├── toxic_news_classification.ipynb # Full EDA, feature engineering, model training
├── requirements.txt              # All dependencies
├── bad_words.txt                 # List of flagged words
├── bert_toxic_model/             # Fine-tuned BERT model (config, weights, tokenizer)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── vocab.txt
│   ├── special_tokens_map.json
│   └── tokenizer_config.json
├── bert_toxic_model_quantized.pt # Quantized model weights
├── jigsaw-toxic-comment-classification-challenge/
│   ├── train.csv/
│   │   └── train.csv
│   ├── test.csv/
│   │   └── test.csv
│   ├── test_labels.csv/
│   │   └── test_labels.csv
│   └── sample_submission.csv/
│       └── sample_submission.csv
└── toxic_comment_classification_env/ 
```

## Minimal Usage

1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the app:
   `streamlit run app.py`
3. Try it out:
   Paste a comment, hit “Classify”, see the probabilities for each toxic class.
