## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 7,
  fig.height = 5
)

## ----eval=FALSE---------------------------------------------------------------
# # Install from GitHub
# remotes::install_github("R-4-Data-Science/FinalProjectmultipathaic")

## ----load-packages, message=FALSE---------------------------------------------
library(multipathaic)
library(lars)
library(caret)

## ----load-data----------------------------------------------------------------
# Load diabetes data
data(diabetes)

str(diabetes)

# Extract response and predictors
y <- diabetes$y
X <- as.data.frame(diabetes$x)

# Display original dimensions
cat("Original dimensions:", nrow(X), "observations,", ncol(X), "predictors\n")

## ----feature-engineering------------------------------------------------------
X_matrix <- unclass(diabetes$x)
X <- as.data.frame(X_matrix)
y <- as.numeric(diabetes$y)

cat("Original predictors:", ncol(X), "\n")

# Step 1: Create ALL second-order terms using formula
formula_expanded <- as.formula(
  paste("~ (", paste(colnames(X), collapse = " + "), ")^2 + I(", 
        paste(colnames(X), "^2", collapse = ") + I("), ")")
)

X_all_terms <- as.data.frame(model.matrix(formula_expanded, data = X))
X_all_terms <- X_all_terms[, -1]  # Remove intercept

# Clean column names
colnames(X_all_terms) <- make.names(colnames(X_all_terms), unique = TRUE)

cat("Total terms created:", ncol(X_all_terms), "\n")

# Step 2: Identify which columns are interactions vs quadratics vs originals
original_vars <- colnames(X)
quadratic_vars <- paste0("I.", colnames(X), ".2.")
interaction_vars <- setdiff(colnames(X_all_terms), c(original_vars, quadratic_vars))

cat("  - Original variables:", length(original_vars), "\n")
cat("  - Quadratic terms:", length(quadratic_vars), "\n")
cat("  - All interactions:", length(interaction_vars), "\n")

# Step 3: Calculate correlations for interactions only
interaction_cors <- sapply(interaction_vars, function(var_name) {
  abs(cor(X_all_terms[[var_name]], y))
})

# Step 4: Select top 40 interactions
top_40_interactions <- names(sort(interaction_cors, decreasing = TRUE)[1:40])

cat("  - Top interactions selected:", length(top_40_interactions), "\n")

# Step 5: Build final dataset with: originals + quadratics + top 40 interactions
X_expanded <- cbind(
  X_all_terms[, original_vars],
  X_all_terms[, quadratic_vars],
  X_all_terms[, top_40_interactions]
)

# Clean up column names
colnames(X_expanded) <- make.names(colnames(X_expanded), unique = TRUE)

cat("\nExpanded feature space:\n")
cat("  - Original variables:", length(original_vars), "\n")
cat("  - Quadratic terms:", length(quadratic_vars), "\n")
cat("  - Top pairwise interactions:", length(top_40_interactions), "\n")
cat("  - Total predictors:", ncol(X_expanded), "\n\n")

# Standardize predictors
X_scaled <- as.data.frame(scale(X_expanded))

cat("Final dimensions:", nrow(X_scaled), "observations ×", ncol(X_scaled), "predictors\n")

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
  K=min(ncol(X), 10),           # Maximum 10 steps
  eps = 1e-6,                   # Minimum AIC improvement
  delta = 2,                    # AIC tolerance for branching
  L = 50,                       # Keep top 50 models per step
  verbose = FALSE
)

print(forest$all_models)

cat("Total models explored:", nrow(forest$all_models), "\n")
cat("Models at final step:", nrow(forest$path_forest$frontiers[[length(forest$path_forest$frontiers)]]), "\n")

## ----stability----------------------------------------------------------------
stab <- stability(
  X = X_train,
  y = y_train,
  family = "gaussian",
  K=min(ncol(X), 10),
  eps = 1e-6,
  delta = 2,
  L = 50,
  B = 50,           # 50 bootstrap resamples
  resample_fraction = 0.8,
  verbose = FALSE
)

print(stab$pi)

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

print(plaus$plausible_models)

cat("Number of plausible models:", nrow(plaus$plausible_models), "\n")
cat("Best AIC:", round(plaus$best_aic, 2), "\n\n")

# Display variable summary
cat("Variable inclusion summary:\n")
print(head(plaus$summary, 15))

# Outcome

library(kableExtra)

best_aic_text <- paste0("Variable Inclusion and Stability Summary (Best AIC = ", 
                        round(plaus$best_aic, 2), ")")

head(plaus$summary, 15) %>%
  kbl(caption = best_aic_text, digits = 3) %>%
  kable_classic(full_width = FALSE, html_font = "Arial")

## ----test-evaluation----------------------------------------------------------
library(knitr)
library(kableExtra)

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
  
  # Create performance summary table
  performance_table <- data.frame(
    Metric = c(
      "Variables Selected",
      "Training RMSE",
      "Test RMSE",
      "Test Correlation (r)",
      "Train R²",
      "AIC (train)"
    ),
    Value = c(
      length(best_model_vars),
      round(train_rmse, 3),
      round(test_rmse, 3),
      round(test_cor, 3),
      round(summary(final_fit)$r.squared, 3),
      round(AIC(final_fit), 3)
    )
  )
  
  # Display table
  kbl(performance_table,
      caption = "Best Model Performance Summary",
      align = c("l", "r"),
      col.names = c("Performance Metric", "Value")) %>%
    kable_styling(bootstrap_options = c("striped", "hover", "condensed"),
                  full_width = FALSE,
                  position = "left") %>%
    row_spec(0, bold = TRUE, color = "white", background = "#3c8dbc") %>%
    column_spec(1, bold = TRUE, width = "15em") %>%
    column_spec(2, width = "10em", color = "#00a65a", bold = TRUE)
  
} else {
  cat("No plausible models found.\n")
}

