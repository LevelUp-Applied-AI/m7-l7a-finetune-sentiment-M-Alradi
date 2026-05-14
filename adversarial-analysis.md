# Adversarial Evaluation Analysis

## Per-hypothesis accuracy

| Hypothesis category | Correct | Total | Accuracy |
|---|---|---|---|
| negation | 4 | 6 | 0.67 |
| lexical_trigger | 1 | 9 | 0.11 |
| domain_shift | 6 | 9 | 0.67 |
| length_extreme | 6 | 8 | 0.75 |
| sarcasm | 0 | 5 | 0.00 |
| other | 5 | 6 | 0.83 |

## Confirmed hypotheses

**Lexical trigger** was the biggest failure as expected — the model only got 1 out of 9 right (11% accuracy, 8 errors). It consistently gets fooled by positive-sounding words like "reliable" even when the sentence is clearly negative (e.g. rows 8–15: "This app is no longer reliable" → predicted neutral or positive instead of negative).

**Sarcasm** was a total failure — 0 out of 5 correct. Every single sarcastic example was predicted as positive (rows 32–36), which makes sense since sarcastic complaints often use positive-sounding words on the surface.

## Refuted hypotheses

**Negation** held up better than expected — 67% accuracy. The model correctly handled many negated phrases like "did not improve" and "did not fix" (rows 1 and 4). Only 2 examples tripped it up (rows 5 and 7).

**Length extreme** also did better than feared — 75% accuracy. The model only failed on 2 long examples (rows 27 and 30), suggesting length alone isn't a major issue.

## What the results reveal about the decision boundary

The model relies heavily on the surface tone of individual words rather than understanding full sentence meaning. When a sentence contains a positive word (like "reliable", "great", "love"), the model leans positive — even if the sentence is sarcastic or negated. This shows up most clearly in lexical_trigger (11%) and sarcasm (0%), where the "feeling" of the words contradicts the actual meaning. The model has essentially learned a bag-of-words shortcut instead of true language understanding.