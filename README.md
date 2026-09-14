# NIR Chemometrics: LASSO, Ridge, and PCA
Regularized regression and dimensionality reduction on near-infrared (NIR) spectroscopy data — a classic high-dimensional, severely multicollinear dataset from the chemometrics field.

What this project does
Predicts glucose/ethanol concentration from 235 NIR wavelength measurements (chemometrics::NIR dataset)
Uses PCA to quantify the dataset's effective dimensionality (~4-6 components explain 90-95% of variance, out of 235 raw predictors)
Compares LASSO and ridge regression, both cross-validated and evaluated on a genuine held-out test set
Interprets why one regularization method outperforms the other, grounded in the PCA result and the geometry of L1 vs. L2 constraint regions
Key result
Model	Test RMSE	Test R²	Predictors retained
LASSO	6.56	0.855	33 of 235
Ridge	11.77	0.532	235 of 235
LASSO substantially outperforms ridge on held-out data — a result well-explained by the PCA diagnostic: with true signal concentrated in a handful of latent dimensions, LASSO's sparse-selection approach matches the underlying structure of the data more closely than ridge's uniform-shrinkage approach.

Tools
R — glmnet, chemometrics, ggplot2, tidyverse

Background
Originally developed as graduate coursework; revised here with a proper held-out test set (the original evaluated LASSO on training data only) and extended with PCA and a ridge comparison to build a complete, coherent analytical narrative.

Files
Chemometrics.Rmd — full analysis (code)
NIR_project_notes.md — technical notes and design decisions
