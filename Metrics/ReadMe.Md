# 📊 Correlation & Prediction Engine

This module evaluates whether sentiment signals correlate with crypto prices and whether they can be used for future predictions.

---

## 🎯 Objective

- Check if sentiment correlates with price  
- Test if sentiment can predict future price movement  
- Evaluate strength and usefulness of sentiment signals  

---

## 🔹 Correlation Analysis

### Pearson’s Correlation Coefficient

Measures linear relationship between sentiment and price.

#### Lag Interpretation:
- **Lag 0** → Sentiment vs same day price  
- **Lag 1** → Sentiment vs next day price  
- **Lag 2+** → Sentiment vs future prices  

👉 Helps identify if sentiment **leads price movement**

---

## 🔹 Granger Causality Test

Tests whether:

> Past sentiment values help predict future price movements  

- Goes beyond correlation  
- Evaluates predictive causality  

---

## 🔹 Binary Classification

Simple prediction logic:

- If sentiment > 0 → predict price goes **UP**  
- If sentiment < 0 → predict price goes **DOWN**  

### Evaluation:
- Compared with actual price direction  
- Metrics include:
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  

---

## 🔹 Macro Correlation Analysis

Analyzes relationship between:

- Macro indicators (e.g., indices, fear-greed)  
- Crypto price movements  

👉 Helps understand external market influence on crypto

---


