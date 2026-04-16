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
