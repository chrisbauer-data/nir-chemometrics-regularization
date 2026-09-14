# ============================================================
# NIR Chemometrics: LASSO, Ridge, and PCA
# Chris Bauer
# ============================================================
# Regularized regression and dimensionality reduction on near-infrared (NIR)
# spectroscopy data. See README.md and NIR_project_notes.md for the full
# write-up and interpretation. This script contains the runnable analysis
# code (also available as Chemometrics.Rmd for the R Markdown / knitted
# version).
# ============================================================

# ---- Load libraries ----
library(glmnet)      # needed for cv.glmnet
library(chemometrics)
library(ggplot2)
library(tidyverse)

data(NIR)

# ---- Data Split into Train and Test Sets ----
X_lasso <- as.matrix(NIR$xNIR)
y_lasso <- NIR$yGlcEtOH[,1]

set.seed(123)
row_num <- nrow(NIR$xNIR)
index_test <- seq(10, row_num, by = 10)

X_train <- X_lasso[-index_test, ]
X_test <- X_lasso[index_test, ]
y_train <- y_lasso[-index_test]
y_test <- y_lasso[index_test]

# ---- Perform PCA ----
pca_result <- prcomp(X_train, scale. = TRUE)

# Variance explained
summary(pca_result)

# Scree plot
plot(pca_result, type = "l", main = "PCA Scree Plot — NIR Wavelengths")

# How many components needed for 90% / 95% variance?
var_explained <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
which(var_explained >= 0.90)[1]
which(var_explained >= 0.95)[1]

# Scores plot: samples projected onto PC1 and PC2
pc_scores <- as.data.frame(pca_result$x[, 1:2])
pc_scores$y <- y_train  # color by response variable to see if PCs separate by outcome

ggplot(pc_scores, aes(x = PC1, y = PC2, color = y)) +
  geom_point(size = 2) +
  scale_color_gradient(low = "blue", high = "red") +
  labs(
    title = "PCA — NIR Spectra Projected onto First Two Principal Components",
    x = paste0("PC1 (", round(var_explained[1] * 100, 1), "% variance)"),
    y = paste0("PC2 (", round((var_explained[2] - var_explained[1]) * 100, 1), "% variance)"),
    color = "Glucose/EtOH\n(response)"
  ) +
  theme_minimal()

# ---- Perform LASSO ----
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 9, standardize = TRUE)

# Lambda selection
cv_lasso$lambda.min
cv_lasso$lambda.1se

lasso_coefs <- coef(cv_lasso, s = "lambda.min")
lasso_coefs

# Held-out test set evaluation
test_preds <- predict(cv_lasso, newx = X_test, s = "lambda.min")

rmse_test <- sqrt(mean((y_test - test_preds)^2))
sst_test <- sum((y_test - mean(y_test))^2)
sse_test <- sum((y_test - test_preds)^2)
r2_test <- 1 - sse_test/sst_test

cat("Test RMSE:", rmse_test, "\n")
cat("Test R-squared:", r2_test, "\n")

# Predicted vs. actual on the held-out test set
plot(y_test, test_preds,
     xlab = "Observed", ylab = "Predicted",
     main = "LASSO fit — held-out test set")
abline(0, 1, col = "red")

plot(cv_lasso)

# Count nonzero coefficients, excluding the intercept
num_nonzero <- sum(lasso_coefs != 0) - 1  # subtract 1 to exclude intercept
cat("Number of nonzero predictors (excluding intercept):", num_nonzero, "\n")

# ---- Perform Ridge ----
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0, nfolds = 9, standardize = TRUE)
best_lambda_ridge <- cv_ridge$lambda.min

ridge_test_preds <- predict(cv_ridge, newx = X_test, s = best_lambda_ridge)

rmse_ridge_test <- sqrt(mean((y_test - ridge_test_preds)^2))
sst_ridge <- sum((y_test - mean(y_test))^2)
sse_ridge <- sum((y_test - ridge_test_preds)^2)
r2_ridge_test <- 1 - sse_ridge/sst_ridge

cat("Ridge Test RMSE:", rmse_ridge_test, "\n")
cat("Ridge Test R-squared:", r2_ridge_test, "\n")
cat("Best Lambda:", best_lambda_ridge, "\n")

plot(y_test, ridge_test_preds,
     xlab = "Observed", ylab = "Predicted",
     main = "Ridge fit — held-out test set")
abline(0, 1, col = "red")

# ============================================================
# Summary
# ============================================================
# LASSO: RMSE = 6.56, R2 = 0.855, 33 of 235 predictors retained
# Ridge: RMSE = 11.77, R2 = 0.532, 235 of 235 predictors retained
#
# LASSO substantially outperforms ridge on held-out data. PCA shows the
# effective dimensionality of the 235-predictor NIR spectrum is only ~4-6
# components (90-95% of variance) -- consistent with LASSO's sparse-selection
# approach matching the true underlying structure of the data more closely
# than ridge's uniform-shrinkage approach.
# ============================================================
