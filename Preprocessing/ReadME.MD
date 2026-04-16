# 📊 NLP Preprocessing Engine

This module prepares raw text data for downstream NLP tasks by cleaning, standardizing, and extracting linguistic features.

---

## 🔹 Preprocessing Steps

### 1. Tokenisation
- Splits the entire post into individual tokens (words)

---

### 2. Lemmatization
- Converts words into their base form  
- Example:
  - "pumping" → "pump"  
  - "running" → "run"

---

### 3. POS Tagging
- Assigns parts of speech to each token  
- Example:
  - noun, verb, adjective, adverb  

---

### 4. Stop Word Removal
- Removes common words that do not add meaningful context  
- Examples:
  - "the", "and", "is"

---

### 5. Slang Normalisation
- Converts crypto-specific slang into standard language  
- Examples:
  - "hodl" → "hold"  
  - "fud" → "fear uncertainty doubt"  

---

### 6. Entity Extraction
- Identifies important entities in text  
- Examples:
  - "Elon Musk" → Person  
  - "SEC" → Organization  

---

### 7. Filtering Short Text
- Posts with very low content are discarded  

---

### 8. TF-IDF Analysis
- Identifies important words in the corpus  
- Gives higher weight to:
  - Frequently occurring words in a document  
  - Rare across all documents  
