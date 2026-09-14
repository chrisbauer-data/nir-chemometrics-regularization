# NIR Chemometrics Regularization — Project Notes

## Overview

**Dataset:** `NIR` dataset from the R `chemometrics` package — near-infrared
spectroscopy measurements across 235 wavelengths, used to predict glucose/ethanol
concentration (`yGlcEtOH`).

**Goal:** Demonstrate regularized regression (LASSO, ridge) and dimensionality
reduction (PCA) on a classic high-dimensional, severely multicollinear dataset.
Originally built as coursework at Pitt; cleaned up and extended for portfolio use.

**Positioning:** R/statistics-depth showcase — companion to the Python-focused
projects (stock tracker) to demonstrate both tool fluency and genuine statistical
reasoning, not just running functions.

---

## Data Split

Train/test split via every 10th observation reserved for testing (`seq(10, n, by=10)`),
`set.seed(123)` for reproducibility. This was a deliberate fix — the original coursework
version evaluated LASSO performance on training data only (in-sample), which is not a
valid measure of generalization. Adding a genuine held-out test set was the first
cleanup pass on this project.

---

## PCA

**Result:** 4 components explain 90% of variance; 6 components explain 95% — out of
235 original predictors.

**Interpretation:** confirms severe multicollinearity in the wavelength data — adjacent
wavelengths carry largely redundant information, so the effective dimensionality of the
dataset is far lower than its raw predictor count suggests. This diagnostic directly
motivates the LASSO vs. ridge comparison below: if true signal lives in ~4-6 latent
dimensions, a sparse-selection method (LASSO) should outperform a shrink-everything
method (ridge).

Scores plot (PC1 vs PC2, colored by response) shows a visible gradient rather than
random scatter — first two components alone capture meaningful structure related to
the outcome.

**Condition number note:** κ(X'X) = κ(X)² — so the condition number of X can be obtained
by taking the square root of the condition number computed on X'X directly. Confirmed
this identity holds (singular values of X'X are the squared singular values of X).

---

## LASSO

`cv.glmnet(X_train, y_train, alpha = 1, nfolds = 9, standardize = TRUE)`

- glmnet generates ~100 candidate lambda values automatically — log-spaced, anchored to
  `lambda_max` (smallest lambda that zeros every coefficient, derived from each
  predictor's correlation with the response). Not an arbitrary grid.
- 9-fold CV evaluates every candidate lambda entirely within the training set; the
  held-out test set is never touched during tuning.
- Model fitting at each lambda uses coordinate descent (fast iterative optimization,
  updates one coefficient at a time); glmnet also reuses computation across adjacent
  lambdas via warm starts.

**Result at `lambda.min`:** 33 of 235 predictors retained (nonzero coefficients,
excluding intercept). Held-out test set: **RMSE = 6.56, R² = 0.855**.

---

## Ridge

`cv.glmnet(X_train, y_train, alpha = 0, nfolds = 9, standardize = TRUE)`

Same CV/lambda-selection mechanism as LASSO, different penalty (L2 vs L1). Ridge
retains all 235 predictors, shrinking coefficients toward zero without eliminating any.

**Result:** Held-out test set: **RMSE = 11.77, R² = 0.532** — substantially
underperforms LASSO.

**Why LASSO beat ridge here (geometric explanation):** L1's constraint region is a
diamond with corners sitting exactly on the coordinate axes; L2's constraint region is
a smooth circle with no corners. Fitting is where the unconstrained solution's
elliptical error contours first touch the constraint boundary as it shrinks — for a
diamond, that first-contact point is disproportionately likely to land on a corner
(where one or more coefficients are exactly zero), while a circle has no preferred
zero-crossing point. This is the direct mechanical reason LASSO produces exact zeros
and ridge doesn't.

**Why this particular result makes sense given PCA:** with 235 highly collinear
predictors compressing into ~4-6 effective dimensions, LASSO's sparsity assumption
matches the true data structure — it correctly identifies and retains the wavelengths
carrying real signal. Ridge's uniform-shrinkage approach instead spreads regularization
thin across many redundant, near-duplicate wavelengths, diluting the fit.

---

## Known Limitations / Honest Caveats

- Train/test split is a single fixed split (every 10th row), not k-fold CV on the
  final held-out evaluation — a single split can be somewhat sensitive to which rows
  land in the test set, especially given the dataset's modest size (NIR spectral
  datasets are typically well under 200 samples).
- LASSO's exact retained-wavelength set is somewhat unstable across different `seed`
  values in real chemometrics practice, due to the extreme multicollinearity — worth
  noting that "which 33 wavelengths" matters less than the fact that ~33 out of 235
  is sufficient, and that the sparsity pattern is directionally consistent with PCA's
  dimensionality estimate.

## Possible Extensions (not built)

- Elastic net (`alpha` between 0 and 1) as a middle ground between LASSO and ridge
- Partial Least Squares (PLS) regression — the more traditional chemometrics-specific
  method for this kind of spectral data, would be a natural comparison point alongside
  LASSO/ridge

## Session Log

- Ported/cleaned original Pitt coursework (condition number check, LASSO)
- Identified two issues in original: no held-out test set validation; condition number
  computed on X'X rather than X directly (resolved via √κ(X'X) = κ(X) identity)
- Added proper train/test split, re-ran LASSO with genuine out-of-sample evaluation
- Added PCA as a diagnostic — quantifies the multicollinearity condition number flagged
- Added ridge regression as a direct comparison to LASSO on the same train/test split
- Added scores plot (PC1 vs PC2) as a visual check on PCA's structure
- Interpreted LASSO-beats-ridge result via PCA's dimensionality finding + L1/L2
  constraint geometry
