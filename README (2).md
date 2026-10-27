# 📰 Fake News Detector — Streamlit UI

## Folder layout
```
fake_news_app/
├── app.py
├── requirements.txt
└── saved_models/          <- downloaded from Colab (/content/saved_models)
    ├── bert_model/
    ├── lstm_model.keras
    ├── gru_model.keras
    └── keras_tokenizer.pkl
```
Any model that is missing is hidden automatically, so you can start with BERT only.

## Option A — Run on your own computer
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Option B — Run inside Google Colab (no downloads)
Upload `app.py` to `/content/`, then run these cells (after the training notebook has saved `/content/saved_models`):

```python
!pip install -q streamlit
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
```
```python
import subprocess, time
subprocess.Popen(["streamlit", "run", "/content/app.py", "--server.port", "8501", "--server.headless", "true"])
time.sleep(8)
!./cloudflared tunnel --url http://localhost:8501
```
Open the `https://....trycloudflare.com` link printed in the output.

## Notes
- `MAX_LEN` (300) and `BERT_MAX_LEN` (128) in `app.py` must match the training notebook.
- Set a different models folder with the sidebar field or `MODEL_DIR=/path streamlit run app.py`.
