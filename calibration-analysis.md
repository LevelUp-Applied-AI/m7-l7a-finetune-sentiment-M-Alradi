# Calibration Analysis

## Reliability diagram interpretation

The bars follow the dashed "perfect calibration" line pretty closely overall, which is a good sign. The interesting part is in the middle — bins around 0.35–0.55 confidence. There the bars sit *above* the diagonal, meaning when the model says "I'm ~46% sure", it's actually right about 50% of the time. So it's slightly under-confident in that zone. On the high end (bin center 0.95), the bar hits 0.958 accuracy against a target of 0.95 — nearly perfect. The model is most honest when it's most sure.

## Expected Calibration Error

ECE = **0.037** (3.7%). As a rough rule of thumb, under 5% is considered well-calibrated. This model is in good shape — if it says it's 80% confident, it's right about 80% of the time in practice. For most production uses you can treat its probability scores as reasonably trustworthy, not just "bigger = more confident" ordinal rankings.

## A specific calibration pattern

The 0.45 bin (center) has 69 predictions but only 46% accuracy — the model is slightly over-confident there. These are almost certainly neutral reviews sitting on the border between neutral and negative/positive. The model was trained on a balanced dataset but neutral is inherently the hardest class (it's a catch-all for "not clearly either"), so it pushes uncertain predictions to ~50% confidence when the true accuracy is a bit lower. The boundary between neutral and the polar classes is fuzzy, and the model hasn't fully learned where that line is.

## A proposed engineering action

Given that the shaky zone is the 0.35–0.55 confidence band (303 predictions, ~20% of the test set), the simplest fix would be **abstention with a human-review queue**: if `max_prob < 0.6`, don't auto-label the review — flag it for a human. Based on the bucket counts, that would catch roughly 303 uncertain predictions per 1,495 test examples. The ones the model *does* auto-label (confidence ≥ 0.6) are correct ~84% of the time on average, which is a much safer bar for production.