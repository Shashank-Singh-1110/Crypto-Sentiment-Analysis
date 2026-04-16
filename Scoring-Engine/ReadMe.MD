## 🔹 Scoring Steps

### 1. VADER Score
- Outputs: `neg`, `neu`, `pos`, `compound`
- Range: `compound ∈ [-1, +1]`

Formula:compound = x / sqrt(x^2 + 15)

### 2. FinBERT Score
- Outputs probabilities: `positive`, `negative`, `neutral`

Derived score: finbert_score = positive - negative

### 3. Composite Score
Combines both models: composite_score = 0.7 * finbert_score + 0.3 * vader_compound

### 4. Engagement Weight
engagement_weight = 1 + log(1 + upvotes) + log(1 + comments)

### 5. Weighted Score
Final impact score: weighted_score = composite_score * engagement_weight

### 6. Z-Score Normalization
Applied on composite score per coin: z = (x - mean) / std

## 📌 Notes
- Weighted score is not normalized
- Z-score is used only for analysis and bias correction
- Falls back to VADER if FinBERT is unavailable

# 📊 Advanced NLP Engine

This module extracts deeper insights from text data using:

- Topic Modeling → What is being discussed  
- Emotion Detection → How users feel  
- Named Entity Recognition (NER) → Important people and organizations  

---

## 🎯 Objective

- **Topic Modeling** → Identify what topics are trending  
- **Emotion Detection** → Understand user sentiment at an emotional level  
- **NER** → Extract key entities (people, organizations, crypto assets)

---

## 🔹 Models Used

- **BERT (via BERTopic)** → Topic Modeling  
- **DistilRoBERTa** → Emotion Detection  

---

## 🔹 Topic Modeling (BERTopic)

### How it works:

1. Text is converted into **vector embeddings** using a transformer (BERT)
2. Similar embeddings are grouped using **HDBSCAN clustering**
3. For each cluster:
   - Important words are extracted
   - These form the **topic representation**
4. Labels are assigned to each cluster
5. Each post is assigned:
   - `topic_id`
   - `topic_label`
   - `topic_probability`

### Notes:
- `-1` → Outliers (irrelevant/noise posts)

---

## 🔹 Emotion Detection (DistilRoBERTa)

### How it works:

1. Text is **tokenized**
2. Model processes tokens using learned contextual representations
3. Outputs probabilities for different emotions
4. Highest probability emotion is selected

### Stored values:
- `emotion_label` → dominant emotion  
- `emotion_score` → confidence  
- `emotion_all` → full distribution  

---

## 🔹 Named Entity Recognition (NER)

Extracts:
- **Organizations**
- **People**
- **Crypto entities**

Combines:
- spaCy model
- Custom crypto and finance dictionaries

---


