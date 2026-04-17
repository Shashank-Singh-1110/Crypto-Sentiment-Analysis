# 📊 Prediction Engine

This module predicts next-day crypto price direction by combining multiple signals and modeling techniques.

---

## 🎯 Objective

To generate price direction predictions using:

- Sentiment signals  
- Momentum  
- Volume  
- Macro indicators  
- Machine Learning  

---

## 🔹 Model Approach

We use a hybrid approach:

- **Logistic Regression (ML model)**
- **Rule-Based Model**

Both models are trained and evaluated, and the **better-performing model is selected**.

---

## 🔹 Feature Engineering

We construct the following features:

- **Sentiment Momentum**  
  Change in sentiment compared to previous day  

- **Volume Ratio**  
  Current post volume vs 7-day average  

- **Model Agreement**  
  Whether VADER and FinBERT agree  

- **Fear & Greed Index**  
  External market sentiment  

- **Target Variable**  
  Next-day price direction (what the model predicts)

---

## 🔹 Optimal Lag Selection

We evaluate different lags:

- **Lag 0** → Same-day sentiment vs price  
- **Lag 1** → Sentiment vs next-day price  
- **Lag 2+** → Sentiment vs future prices  

👉 The lag with the **highest correlation** is selected.

---

## 🔹 Contrarian Behavior

If correlation is negative:

- Sentiment signal is inverted  
- Example:
  - Bullish sentiment → price may fall  
  - Bearish sentiment → price may rise  

---

## 🔹 Model Selection

- Train both:
  - Logistic Regression  
  - Rule-based model  

- Evaluate performance  
- Select the model with **better results**

---

## 🔹 Evaluation Metrics

Models are evaluated using:

- **Accuracy**  
- **Baseline Accuracy**  
- **F1 Score**  

👉 Ensures model performs better than random prediction

---

## 🔹 Prediction Process

1. Take latest available data  
2. Use engineered features  
3. Apply trained model  
4. Generate:
   - **Prediction** (UP / DOWN)  
   - **Probability score**  

---

## 🔹 Confidence Estimation

Confidence is based on:

- **Z-score (sentiment strength)**  
- **Model probability output**
