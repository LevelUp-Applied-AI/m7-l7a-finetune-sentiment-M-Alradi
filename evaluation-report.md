# Lab Evaluation Report

## Dataset

The training data is drawn from the AARSynth app reviews dataset, curated across 9 apps (Bitmoji, AccuWeather, Adobe Acrobat Reader, Adobe Lightroom, Booking.com, Forest, Slack, UC Browser, BBM) with star ratings mapped to three sentiment classes: 0 = negative, 1 = neutral, 2 = positive. The full training CSV contains 7,472 reviews balanced across (app, class) buckets; after an 80/20 split (seed = 42) this yields ~5,977 training examples and ~1,495 test examples, with roughly equal class representation (positive: 533, negative: 499, neutral: 463 in the test split).

## Model and Hyperparameters

| Parameter | Value |
|---|---|
| Backbone | `distilbert-base-uncased` |
| Number of labels | 3 |
| Learning rate | 5e-5 |
| Epochs | 2 |
| Batch size (train & eval) | 8 |
| Max sequence length | 128 |
| Seed | 42 |
| Training time (wall-clock) | ~18 min on CPU (Intel Core i7, no GPU) |

## Metrics on the Test Split

**Aggregate:**

| Metric | Value |
|---|---|
| Accuracy | 0.6321 |
| Macro-F1 | 0.6304 |

**Per class:**

| Class | F1 | Precision | Recall |
|---|---|---|---|
| Negative | 0.7046 | 0.7018 | 0.7074 |
| Neutral | 0.4851 | 0.4627 | 0.5097 |
| Positive | 0.7015 | 0.7386 | 0.6679 |

The neutral class is clearly the weakest link — it sits roughly 20 points below negative and positive on F1. This is expected: neutral reviews are short, factual, or contain a mix of positive and negative cues that the model struggles to balance.

## Confusion Matrix

Rows = true label, columns = predicted label.

|  | negative | neutral | positive |
|---|---|---|---|
| **negative** | 353 | 129 | 17 |
| **neutral** | 118 | 236 | 109 |
| **positive** | 35 | 143 | 355 |

The largest off-diagonal cells are neutral→negative (118) and neutral→positive (109), confirming the model is systematically uncertain about the middle class and tends to collapse it into one of the polar classes. Negative→positive confusion is rare (17), which is reassuring — the model rarely makes a sign-flip error on clearly polar reviews.

## Three Qualitative Error Examples

### Error 1 — Neutral predicted as Negative
**Text:** `"my company forces me to use this app."`  
**Gold label:** neutral  
**Predicted label:** negative  
**Predicted probability for gold label (prob_neutral):** 0.033  

This sentence expresses a factual constraint rather than a complaint, but the word *forces* is a strong negative cue in general language ("forced to do X" often implies unwillingness). The model almost certainly overweighted that single word and ignored the absence of any explicit complaint about the app itself. A human reader recognises the neutral, matter-of-fact tone; the model treats the sentiment-laden verb as the dominant signal.

---

### Error 2 — Positive predicted as Negative (mixed, long review)
**Text:** `"it's like <number> stars for the app, <number> star for new booking assistant. god damn it, how the hell can i just answer the message i got from a property???!!! why do i need to see <number> questions of an assistant instead, that i can not delete and that just mess up all the dialog! hate it. really hate this feature! do like everything else though."`  
**Gold label:** positive  
**Predicted label:** negative  
**Predicted probability for gold label (prob_positive):** 0.019  

This is a classic mixed-sentiment review: the user likes the overall app (4–5 stars worth) but intensely dislikes one specific new feature. The rant dominates the text — *god damn it*, *hate it*, *really hate this feature* — while the positive qualifier ("do like everything else though") is brief and buried at the end. DistilBERT processes the full sequence but the sheer emotional intensity and volume of negative language in the middle of the review swamps the short positive closing clause.

---

### Error 3 — Negative predicted as Positive (genuinely hard for humans)
**Text:** `"excellent pdf viewer rendered unusable by the inability to jump to a page and by the fact that it doesn't remember where you were or bookmarks."`  
**Gold label:** negative  
**Predicted label:** positive  
**Predicted probability for gold label (prob_negative):** 0.011  

This example is genuinely difficult. The review opens with *excellent*, a strongly positive word, and the star rating that produced the negative label (1–2 stars) is not visible to the model — only the text is. Even a careful human reader might initially assess this as mixed or even positive without re-reading carefully. The reviewer uses a classic "X but Y" structure where the second clause overturns the first, but the opening adjective is so emphatic that the model commits to positive with very high confidence. This calibrates expectations: for reviews that use sarcasm, backhanded compliments, or "bait-and-switch" phrasing, sentiment classification from text alone has a hard ceiling.

---

## Hugging Face Hub Model URL

https://huggingface.co/alradi/m7-app-review-sentiment
