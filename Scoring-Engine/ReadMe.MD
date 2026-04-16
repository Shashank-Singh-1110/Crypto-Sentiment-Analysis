2. FinBERT Sentiment Analysis
We use FinBERT, a transformer-based model fine-tuned for financial sentiment.
Outputs:
positive probability
negative probability
neutral probability
Derived Score:
finbert_score
=
P
(
positive
)
−
P
(
negative
)
finbert_score=P(positive)−P(negative)
Range: [-1, +1]
Captures contextual and financial sentiment more accurately than rule-based methods
🔹 3. Composite Sentiment Score
We combine VADER and FinBERT to get a more robust sentiment signal.
Formula:
composite_score
=
0.7
×
finbert_score
+
0.3
×
vader_compound
composite_score=0.7×finbert_score+0.3×vader_compound
Rationale:
70% weight to FinBERT → contextual understanding
30% weight to VADER → lexical + domain-specific signals
👉 This represents the final sentiment before engagement adjustment
🔹 4. Engagement Weight
To capture the importance and reach of a post, we incorporate engagement metrics.
Formula:
engagement_weight
=
1
+
log
⁡
(
1
+
upvotes
)
+
log
⁡
(
1
+
comments
)
engagement_weight=1+log(1+upvotes)+log(1+comments)
Purpose:
Boosts influence of highly engaged posts
Uses logarithmic scaling to prevent extreme outliers
🔹 5. Weighted Score (Final Impact Score)
The final score reflects both sentiment and engagement.
Formula:
weighted_score
=
composite_score
×
engagement_weight
weighted_score=composite_score×engagement_weight
Interpretation:
High sentiment + high engagement → high influence
Low sentiment + high engagement → still impactful
Used for analysis, ranking, and downstream insights
🔹 6. Z-Score Normalization
To correct for bias and skew across different coins, we normalize sentiment scores.
Formula:
z
=
x
−
μ
σ
z= 
σ
x−μ
​	
 
Where:
x = composite score
μ = mean score (per coin)
σ = standard deviation (per coin)
Purpose:
Removes coin-specific sentiment bias
Standardizes scores for fair comparison
Helps identify relative sentiment strength
🔹 7. Sentiment Interpretation
Positive z-score → Bullish sentiment
Negative z-score → Bearish sentiment
Near zero → Neutral sentiment
