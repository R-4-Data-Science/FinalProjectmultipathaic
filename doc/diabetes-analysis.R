## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 7,
  fig.height = 5
)

## ----load-packages, message=FALSE---------------------------------------------
library(multipathaic)
library(lars)
library(caret)

## ----load-data----------------------------------------------------------------
# Load diabetes data
data(diabetes)

head(diabetes)

# Extract response and predictors
y <- diabetes$y
X <- as.data.frame(diabetes$x)

# Display original dimensions
cat("Original dimensions:", nrow(X), "observations,", ncol(X), "predictors\n")

## ----feature-engineering------------------------------------------------------
# Create formula for second-order terms
formula_expanded <- as.formula(
  paste("~ (", paste(colnames(X), collapse = " + "), ")^2 + I(", 
        paste(colnames(X), "^2", collapse = ") + I("), ")")
)

# Generate expanded feature matrix
X_expanded <- as.data.frame(model.matrix(formula_expanded, data = X))
X_expanded <- X_expanded[, -1]  # remove intercept

# Clean column names to avoid formula issues
colnames(X_expanded) <- make.names(colnames(X_expanded), unique = TRUE)

cat("Expanded dimensions:", ncol(X_expanded), "predictors\n")

# Standardize predictors
X_scaled <- as.data.frame(scale(X_expanded))

## ----train-test-split---------------------------------------------------------
set.seed(123)
train_idx <- createDataPartition(y, p = 0.7, list = FALSE)

X_train <- X_scaled[train_idx, ]
X_test <- X_scaled[-train_idx, ]
y_train <- y[train_idx]
y_test <- y[-train_idx]

cat("Training set:", nrow(X_train), "samples\n")
cat("Test set:", nrow(X_test), "samples\n")

## ----build-paths--------------------------------------------------------------
forest <- build_paths(
  X = X_train,
  y = y_train,
  family = "gaussian",
  K = 10,           # Maximum 10 steps
  eps = 1e-6,       # Minimum AIC improvement
  delta = 2,        # AIC tolerance for branching
  L = 50,           # Keep top 50 models per step
  verbose = FALSE
)

cat("Total models explored:", nrow(forest$all_models), "\n")
cat("Models at final step:", nrow(forest$path_forest$frontiers[[length(forest$path_forest$frontiers)]]), "\n")

## ----stability----------------------------------------------------------------
stab <- stability(
  X = X_train,
  y = y_train,
  family = "gaussian",
  K = 10,
  eps = 1e-6,
  delta = 2,
  L = 50,
  B = 50,           # 50 bootstrap resamples
  resample_fraction = 0.8,
  verbose = FALSE
)

# Display top stable variables
cat("\nTop 10 most stable variables:\n")
print(round(head(stab$pi, 10), 3))

## ----plausible-models---------------------------------------------------------
plaus <- plausible_models(
  forest = forest,
  pi = stab$pi,
  Delta = 2,
  tau = 0.6,
  verbose = FALSE
)

cat("Number of plausible models:", nrow(plaus$plausible_models), "\n")
cat("Best AIC:", round(plaus$best_aic, 2), "\n\n")

# Display variable summary
cat("Variable inclusion summary:\n")
print(head(plaus$summary, 15))

## ----test-evaluation----------------------------------------------------------
if (nrow(plaus$plausible_models) > 0) {
  # Extract variables from best model
  best_model_vars <- plaus$plausible_models$model[[1]]
  
  # Fit on training data
  train_df <- data.frame(y = y_train, X_train[, best_model_vars, drop = FALSE])
  final_fit <- lm(y ~ ., data = train_df)
  
  # Predict on test set
  test_df <- X_test[, best_model_vars, drop = FALSE]
  y_pred <- predict(final_fit, newdata = test_df)
  
  # Compute metrics
  test_rmse <- sqrt(mean((y_test - y_pred)^2))
  train_pred <- predict(final_fit)
  train_rmse <- sqrt(mean((y_train - train_pred)^2))
  test_cor <- cor(y_test, y_pred)
  
  cat("\n=== Best Model Performance ===\n")
  cat("Variables selected:", length(best_model_vars), "\n")
  cat("Variable names:", paste(best_model_vars, collapse = ", "), "\n\n")
  cat("Training RMSE:", round(train_rmse, 3), "\n")
  cat("Test RMSE:", round(test_rmse, 3), "\n")
  cat("Test Correlation:", round(test_cor, 3), "\n")
  cat("R-squared (train):", round(summary(final_fit)$r.squared, 3), "\n")
}

