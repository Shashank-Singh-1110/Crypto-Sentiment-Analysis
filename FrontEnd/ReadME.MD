# 🌐 Flask Backend (Main Server)

This module acts as the backend server for the Crypto Sentiment Dashboard.  
It provides API endpoints to serve data, run predictions, and control the pipeline.

---

## 🎯 Objective

- Serve processed data to frontend (HTML/CSS/JS)
- Provide APIs for sentiment, prices, predictions, and analysis
- Integrate LLM (Llama3 via Ollama) for insights
- Trigger full pipeline refresh

---

## 🔹 Core Functionality

- Flask-based API server  
- Connects to SQLite database (`project.db`)  
- Loads trained ML models for predictions  
- Integrates Llama3 for AI-generated insights  
- Handles full pipeline execution  

---

## 🔹 API Endpoints

### 📊 `/api/stats`
- Returns overall statistics:
  - Total posts  
  - Date range  
  - Processed data counts  
  - Topics discovered  

---

### 📈 `/api/sentiment/<coin>`
- Returns time-series sentiment data:
  - VADER  
  - FinBERT  
  - Composite  
  - Z-score  
  - Weighted score  

---

### 💰 `/api/prices/<coin>`
- Returns daily price data:
  - Open, High, Low, Close  
  - Volume  
  - Daily returns  

---

### 🧪 `/api/ablation`
- Returns ablation study results  
- Compares performance of different sentiment models  

---

### 🔗 `/api/correlations`
- Returns:
  - Pearson correlation  
  - Granger causality  
- Shows relationship between sentiment and price  

---

### 🔮 `/api/predict/<coin>`
- Returns prediction for next-day price:
  - Direction (UP/DOWN)  
  - Probability  
  - Confidence  
  - Contributing factors  

---

### 🔄 `/api/refresh`
- Triggers full pipeline execution:
  - Data collection  
  - Preprocessing  
  - Sentiment scoring  
  - Advanced NLP  
  - Correlation analysis  

---

### 🧠 `/api/insight`
- Uses Llama3 (via Ollama)  
- Generates AI-based insights from:
  - Sentiment data  
  - Price trends  
  - Emotion distribution  

---

## 🔹 Frontend

- Built using:
  - HTML  
  - CSS  
  - JavaScript  

- Consumes API endpoints to display:
  - Charts  
  - Predictions  
  - Insights  

