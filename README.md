# 📰 Fake News Detection Using Deep Learning

A deep-learning system that classifies news articles as **REAL** or **FAKE**, with a Streamlit web app for trying it out. Three models are trained and compared on the same data: a bidirectional **LSTM**, a bidirectional **GRU**, and a fine-tuned **BERT**.

## Why this project

Fake news spreads quickly on social media and can cause real harm. This project explores how well neural networks can learn the *writing patterns* that separate misleading articles from genuine reporting, to support fact-checking work. It is a classifier of text style, not a fact-checker (see [Limitations](#limitations)).

## Features

- Three models trained and compared: LSTM, GRU, fine-tuned BERT
- Pretrained GloVe embeddings for the recurrent models
- Full evaluation: accuracy, precision, recall, F1, confusion matrices, ROC curves
- Streamlit app with single-model and **Compare all** modes, example texts, confidence display and a session history
- Everything in one PyTorch notebook that runs end to end on a free Colab GPU

## Results

Evaluated on the same held-out test set (909 articles; FAKE is the positive class):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| LSTM | 0.892 | 0.863 | 0.935 | 0.898 |
| GRU | 0.904 | 0.899 | 0.913 | 0.906 |
| **BERT** | **0.965** | **0.984** | 0.946 | **0.965** |

BERT is clearly the strongest model. The LSTM/GRU gap is small enough to be run-to-run noise. These numbers come from a single training run and one random split, so expect a point or two of variation if you retrain.

## Dataset

The public **"Fake or Real News"** dataset: 6,335 labelled articles, roughly balanced between FAKE and REAL. After removing missing and duplicate rows, 6,060 remain, split 70/15/15 (4,242 train / 909 validation / 909 test) with stratification. The notebook downloads it automatically from a public GitHub mirror, so no manual download or API key is needed.

## Models

| | LSTM / GRU | BERT |
|---|---|---|
| Input | Cleaned text (lowercased, stopwords removed) | Raw text |
| Representation | GloVe 100d embeddings, 20,000-word vocabulary, 300 tokens max | `bert-base-uncased` WordPiece tokens, 128 tokens max |
| Architecture | 2-layer bidirectional recurrent network, hidden size 64, small dense head with dropout | Pretrained BERT with a classification head |
| Training | Adam (lr 1e-3), early stopping on validation loss | AdamW (lr 2e-5), linear schedule, 2 epochs, batch size 16 |

All models are implemented in PyTorch.

## Project structure

```
fake_news_app/
├── app.py                                   # Streamlit web app
├── requirements.txt                         # Python dependencies
├── Fake_News_Detection_Deep_Learning.ipynb  # Training and evaluation notebook (Colab)
├── README.md
└── saved_models/
    ├── bert_model/                          # Fine-tuned BERT (large file, stored with Git LFS)
    ├── lstm_model.pt
    ├── gru_model.pt
    ├── vocab.pkl                            # Vocabulary for LSTM/GRU
    └── config.json                          # Sequence lengths used in training
```

## Getting started

### 1. Run the web app locally

You need Python 3.9+ and [Git LFS](https://git-lfs.com) (the BERT weights are stored with it).

```bash
git lfs install
git clone https://github.com/lucifer55555555/fake_news_app.git
cd fake_news_app

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Open the local URL that Streamlit prints. Pick a model in the sidebar, paste an article and click **Analyze**. Any model whose files are missing is hidden automatically, so the app also works with only some of them.

> If you download the repo as a ZIP instead of cloning, the BERT weights will be tiny pointer files and BERT will fail to load. Use `git clone` with Git LFS installed.

You can point the app at a different models folder with the sidebar field or with `MODEL_DIR=/path/to/models streamlit run app.py`.

### 2. Retrain the models in Google Colab

1. Open `Fake_News_Detection_Deep_Learning.ipynb` in [Google Colab](https://colab.research.google.com).
2. Choose **Runtime → Change runtime type → GPU (T4)**.
3. Choose **Runtime → Run all** once, without restarting in between.
4. At the end the notebook writes `saved_models/` and a `saved_models.zip`. Download the zip and unzip it next to `app.py`.

Rough timings on a T4: GloVe download 1 to 2 minutes, LSTM and GRU a few minutes each, BERT about 10 minutes.

## How the app works

- **BERT** takes the raw text, tokenizes it with its own tokenizer and returns the probability of FAKE.
- **LSTM / GRU** clean the text the same way as in training, map words to vocabulary ids, then return the probability of FAKE.
- A prediction is shown as FAKE when that probability is at least 0.5. Results under 65% confidence are marked as low confidence.
- **Compare all** runs every available model and uses the average probability for the headline verdict.

## Limitations

- **Style, not truth.** The models detect writing patterns learned from a labelled dataset. They cannot check whether a claim is factually correct, and a false claim written in calm, formal language can be classified as REAL.
- **Narrow training data.** The dataset is mostly 2016-era US political news. Accuracy may drop on other topics (entertainment, science, sport) and time periods, and the models can be confidently wrong there.
- **Short inputs.** The models were trained on full articles, so single headlines give noisier predictions.
- **Single run.** Reported scores come from one split and one seed.

Always cross-check important claims with trusted fact-checking sources.

## Tech stack

Python, PyTorch, Hugging Face Transformers, NLTK, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, Streamlit. GloVe embeddings from Stanford NLP.

## Ideas for future work

- Word-level explanations (LIME or SHAP) to show which words drive a prediction
- Training on larger and more recent datasets (for example LIAR or ISOT), or combining several
- More BERT epochs and multiple seeds with averaged results
- Fetching article text from a URL in the app
- Multi-class labels such as satire, misleading or biased, instead of binary fake/real

## Acknowledgements

- "Fake or Real News" dataset, via a public GitHub mirror
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/), Stanford NLP
- [BERT](https://arxiv.org/abs/1810.04805) and the Hugging Face Transformers library
