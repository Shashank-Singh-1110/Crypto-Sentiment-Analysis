Sentiment Scoring data after applying:
(i) Vader- Which takes sentiment in account for scoring.
(ii) FinBERT - Which takes finanical knowledge into account for scoring.
(iii) Composite Score - Which takes both of the model VADER and FinBERT into scoring.
(iv) Engagement Weight - Takes reach and how popular the post into adding a boost.
(v) Weight Score - Multiplication of Composite and Engagment, Final boosted score.
(vi) Z-Score - Since the data aligns towards positive side to normalise it we use z-score. Give actual unboosted score.
