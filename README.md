# 🔮 Crypto Sentiment Analysis & Price Prediction

**Does Reddit sentiment predict cryptocurrency price movements?**

An end-to-end NLP pipeline that collects Reddit data, applies dual-model sentiment analysis (VADER + FinBERT), and statistically tests whether social media sentiment can predict short-term crypto price movements.
---

## 📊 Key Findings

| Finding | Result | Significance |
|---------|--------|-------------|
| FinBERT vs VADER | r=0.27 vs r=0.14 | FinBERT captures financial context VADER misses |
| Granger Causality (MARKET → BTC) | F=3.29, lag-2 | **p=0.042 — statistically significant** |
| MARKET sentiment | r=-0.27 at lag-2 | Contrarian signal — bearish Reddit predicts price UP |
| BTC vs S&P 500 | r=0.52 | Macro factors are the dominant price driver |
| BTC prediction accuracy | 58.1% vs 57.0% baseline | Beats baseline with rule-based ensemble |
| SOL prediction accuracy | 57.6% vs 51.5% baseline | +6.1% improvement over baseline |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
├──────────────────┬──────────────────┬───────────────────────────┤
│ Module 1         │ Module 2         │ Module 2B                 │
│ Reddit Posts     │ Crypto Prices    │ Macro Indicators          │
│ Arctic Shift API │ Binance REST API │ yfinance + Alternative.me │
│ 75,714 posts     │ 6 coins OHLCV   │ Oil, DXY, Gold, VIX, S&P │
│ 11 subreddits    │ yfinance backup  │ Fear & Greed Index        │
└────────┬─────────┴────────┬─────────┴─────────────┬─────────────┘
         │                  │                       │
         ▼                  │                       │
┌──────────────────┐        │                       │
│ Module 3         │        │                       │
│ NLP Preprocessing│        │                       │
│ spaCy + 38 slang │        │                       │
│ TF-IDF bigrams   │        │                       │
└────────┬─────────┘        │                       │
         │                  │                       │
         ▼                  │                       │
┌──────────────────┐        │                       │
│ Module 4         │        │                       │
│ Sentiment Scoring│        │                       │
│ VADER + FinBERT  │        │                       │
│ Composite + Zscore│       │                       │
└────────┬─────────┘        │                       │
         │                  │                       │
         ▼                  │                       │
┌──────────────────┐        │                       │
│ Module 5         │        │                       │
│ Advanced NLP     │        │                       │
│ BERTopic+Emotions│        │                       │
│ NER + Entities   │        │                       │
└────────┬─────────┘        │                       │
         │                  │                       │
         ▼                  ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Module 6 — Correlation Engine                                   │
│ Pearson (lagged) · Granger Causality · Binary Classification    │
│ Ablation Study (5 variants × 7 coins)                           │
│ Macro Correlation Analysis                                      │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────────────────────────┐
│ Module 8         │     │ Module 7 — Flask Dashboard           │
│ Prediction Engine│────▶│ 11 API endpoints · 6 pages           │
│ Logistic Reg +   │     │ Chart.js · Llama 3 via Ollama        │
│ Rule Ensemble    │     │ Real-time prediction & AI insights   │
└──────────────────┘     └──────────────────────────────────────┘
```

All modules share a single **SQLite database** (`data/project.db`) for seamless JOIN operations.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.12 |
| **NLP** | spaCy, HuggingFace Transformers, NLTK |
| **Sentiment** | VADER (+ 52 crypto terms), FinBERT (ProsusAI/finbert) |
| **Topic Modeling** | BERTopic (sentence-transformers + HDBSCAN) |
| **Emotions** | DistilRoBERTa (j-hartmann/emotion-english-distilroberta-base) |
| **ML** | scikit-learn (Logistic Regression, StandardScaler, TimeSeriesSplit) |
| **Statistics** | scipy (Pearson, Granger), statsmodels |
| **Data** | SQLite, pandas, numpy |
| **APIs** | Arctic Shift (Reddit), Binance REST, yfinance, Alternative.me |
| **Backend** | Flask |
| **Frontend** | Vanilla HTML/CSS/JS, Chart.js |
| **LLM** | Llama 3 via Ollama |
| **Database** | SQLite (single shared file) |

---

## 📁 Project Structure

```
crypto_sentiment/
├── Data_Collector.py          # Module 1: Reddit scraping (Arctic Shift + PullPush)
├── price_collector.py         # Module 2: Crypto OHLCV (Binance + yfinance)
├── macro_collector.py         # Module 2B: Macro indicators
├── nlp_preprocessor.py        # Module 3: spaCy + slang normalisation + TF-IDF
├── sentiment_scorer.py        # Module 4: VADER + FinBERT + composite + z-score
├── advanced_nlp.py            # Module 5: BERTopic + emotions + NER
├── Correlation.py             # Module 6: Pearson, Granger, ablation study
├── Prediction.py              # Module 8: Logistic regression + rule ensemble
├── app.py                     # Module 7: Flask backend (11 API endpoints)
├── static/
│   └── index.html             # Dashboard frontend (6 pages, Chart.js)
├── data/
│   ├── project.db             # Shared SQLite database
│   ├── correlation/           # Pearson, Granger, ablation CSVs
│   ├── models/                # Trained prediction models (.pkl)
│   └── predictions/           # Model evaluation results
├── exports/                   # CSV exports of Reddit data
├── logs/                      # Module execution logs
└── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip
- Ollama (optional, for AI Insights page)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/crypto-sentiment.git
cd crypto-sentiment

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Run the Pipeline

```bash
# Step 1: Collect Reddit data (takes ~30 min for 90 days)
python Data_Collector.py --days 90

# Step 2: Collect crypto prices
python price_collector.py --days 90
# If Binance is blocked in your region:
python price_collector.py --days 90 --source yfinance

# Step 3: Collect macro indicators
python macro_collector.py --days 90

# Step 4: Preprocess text (5-10 min)
python nlp_preprocessor.py

# Step 5: Score sentiment (~30-45 min with FinBERT on CPU)
python sentiment_scorer.py

# Step 6: Advanced NLP — topics, emotions, NER (~15-20 min)
python advanced_nlp.py

# Step 7: Run correlation analysis (~1 min)
python Correlation.py

# Step 8: Train prediction models (~10 sec)
python Prediction.py

# Step 9: Launch the dashboard
python app.py
```

Open **http://localhost:5000** in your browser.

### Optional: Enable AI Insights

```bash
# Install and start Ollama
ollama serve
ollama pull llama3

# The AI Insights page will now respond to queries
```

---

## 📈 Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | Stats strip, market sentiment chart, volume donut, per-coin sentiment bars |
| **Sentiment** | Interactive coin selector, VADER/FinBERT/Composite lines, volume bars, divergence chart |
| **Predict** | Tomorrow's price direction, probability gauge, contributing factors, model accuracy |
| **Research** | Ablation study table, accuracy charts, lag-1 correlation charts, key findings |
| **Explore** | Emotion doughnut, NER entity table, BERTopic topic trends |
| **AI Insights** | Chat with Llama 3 about your data, quick-ask buttons, coin context selector |

---

## 🔬 Methodology

### Sentiment Scoring Pipeline

```
Reddit Post
    │
    ├──→ VADER (rule-based, 52 crypto terms) ──→ compound score (-1 to +1)
    │
    ├──→ FinBERT (transformer, financial text) ──→ score (-1 to +1)
    │
    ▼
Composite = 0.7 × FinBERT + 0.3 × VADER
    │
    ▼
Engagement Weight = 1 + log1p(upvotes) + log1p(comments)
    │
    ▼
Weighted Score = Composite × Engagement Weight
    │
    ▼
Z-Score = (daily_avg - coin_mean) / coin_std
```

### Ablation Study Results

| Variant | BTC Accuracy | Baseline | Beats? |
|---------|-------------|----------|--------|
| VADER only | 44.8% | 56.0% | No |
| FinBERT only | 52.0% | 56.0% | No |
| Composite | 54.4% | 56.0% | No |
| **Z-score corrected** | **59.2%** | **56.0%** | **Yes** |
| Full weighted | 49.6% | 56.0% | No |

### Prediction Model

4-feature logistic regression + rule-based ensemble:
- `composite` — combined VADER + FinBERT score
- `zscore` — how unusual today's sentiment is
- `composite_momentum` — sentiment change from yesterday
- `model_agreement` — do VADER and FinBERT agree?

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/stats` | GET | Overall dashboard statistics |
| `/api/sentiment/<coin>` | GET | Daily sentiment scores |
| `/api/prices/<coin>` | GET | OHLCV price data |
| `/api/predict/<coin>` | GET | ML prediction for tomorrow |
| `/api/ablation` | GET | Ablation study results |
| `/api/emotions/<coin>` | GET | 7-class emotion distribution |
| `/api/ner/top` | GET | Top named entities |
| `/api/topics/trends` | GET | BERTopic trend data |
| `/api/correlations` | GET | Pearson, Granger, heatmap |
| `/api/insight` | POST | LLM analysis via Ollama |
| `/api/refresh` | POST | Re-run entire pipeline |

---

## 📋 Requirements

```
flask
pandas
numpy
scipy
statsmodels
scikit-learn
spacy
transformers
torch
vaderSentiment
bertopic
sentence-transformers
hdbscan
requests
yfinance
```

---

## 🔑 Key Design Decisions

**Why Arctic Shift over PRAW?**
Reddit's official API requires paid credentials and has strict rate limits. Arctic Shift is free, open-source, and allows historical queries — making this research fully reproducible.

**Why SQLite over PostgreSQL?**
SQLite is file-based and zero-configuration. One database file that every module reads from and writes to. No server to install, no credentials to manage. Perfect for a research project where reproducibility matters.

**Why Logistic Regression over LSTM?**
With 90 daily observations, deep learning would severely overfit. We confirmed this empirically — 12 features produced 45% accuracy (below coin-flip). Reducing to 4 features with strong regularisation achieved 58.1%. Simpler models generalise better on small datasets.

**Why 0.7 FinBERT + 0.3 VADER?**
FinBERT achieves r=0.27 correlation with BTC returns versus r=0.14 for VADER. FinBERT deserves higher weight. But VADER catches crypto slang (HODL, mooning, rekt) that FinBERT has never seen. The 30% VADER contribution fills this gap.

**Why Z-score normalisation?**
Different coin communities have different baseline sentiments. r/dogecoin is inherently more optimistic than r/CryptoCurrency. Z-scoring centres each coin's distribution so +0.05 sentiment means the same thing for all coins. The ablation study proved this is the only variant that consistently beats baseline.

---

## 📄 Citation

If you use this project in academic work, please cite:

```
Singh, S. (2026). Crypto Sentiment Analysis: NLP-Based Analysis of Reddit 
Social Sentiment and Cryptocurrency Price Movements. Final Year Project.
```

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Arctic Shift](https://arctic-shift.photon-reddit.com/) — Reddit data archive
- [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) — Financial sentiment model
- [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) — Emotion classification
- [Alternative.me](https://alternative.me/crypto/fear-and-greed-index/) — Crypto Fear & Greed Index
- [Ollama](https://ollama.ai/) — Local LLM inference
