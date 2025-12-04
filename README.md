# multipathaic

Multi-Path Stepwise Selection with AIC for Linear and Logistic Regression

## Overview

`multipathaic` implements a multi-path forward selection algorithm using the Akaike Information Criterion (AIC) for both linear (Gaussian) and logistic (binomial) regression models. Unlike traditional stepwise methods that follow a single greedy path, this approach explores multiple competitive model paths simultaneously by retaining near-optimal models at each step.

### Key Features

- **Multi-path exploration**: Maintains multiple competitive models at each forward selection step
- **Stability assessment**: Bootstrap resampling to identify reliable predictors
- **Plausible model selection**: Combines AIC-based filtering with stability thresholds
- **Dual family support**: Works with both Gaussian (linear) and binomial (logistic) regression
- **Diagnostic tools**: Confusion matrix and performance metrics for classification models

### Algorithms Implemented

1. **`build_paths()`** — Multi-path forward selection using AIC with branching
2. **`stability()`** — Resampling-based stability estimation with model-set proportions
3. **`plausible_models()`** — AIC tolerance filtering combined with average stability thresholds
4. **`multipath_aic()`** — Complete pipeline combining all three steps
5. **`confusion_metrics()`** — Performance evaluation for logistic regression models

## Installation

Install the package directly from GitHub:
```r
# Install remotes if not already installed
install.packages("remotes")

# Install multipathaic
remotes::install_github("R-4-Data-Science/FinalProjectmultipathaic")
```

## Quick Start

### Linear Regression Example (Gaussian)
```r
library(multipathaic)

# Generate synthetic data
set.seed(123)
n <- 150; p <- 10
X <- as.data.frame(matrix(rnorm(n*p), n, p))
names(X) <- paste0("x", 1:p)
beta <- c(2, -1.5, 0, 0, 1, rep(0, p-5))
y <- as.numeric(as.matrix(X) %*% beta + rnorm(n, 1))

# Step 1: Build multi-path forest
forest <- build_paths(
  X = X, 
  y = y, 
  family = "gaussian",
  K = 10,
  eps = 1e-6,
  delta = 2,
  L = 50
)

# Step 2: Compute stability across resamples
stab <- stability(
  X = X,
  y = y,
  family = "gaussian",
  B = 50,
  K = 10,
  eps = 1e-6,
  delta = 2,
  L = 50
)

# Step 3: Select plausible models
plaus <- plausible_models(
  forest = forest,
  pi = stab$pi,
  Delta = 2,
  tau = 0.6
)

# View results
print(plaus$summary)
```

**Or use the all-in-one wrapper:**
```r
# Complete pipeline in one function
result <- multipath_aic(
  X = X,
  y = y,
  family = "gaussian",
  K = 10,
  B = 50,
  Delta = 2,
  tau = 0.6
)

# Access components
result$forest        # Multi-path search results
result$stab          # Stability scores
result$plaus         # Plausible models
```

### Logistic Regression Example (Binomial)
```r
library(multipathaic)

# Generate binary outcome data
set.seed(42)
n <- 200; p <- 8
X <- as.data.frame(matrix(rnorm(n*p), n, p))
names(X) <- paste0("x", 1:p)

# True model: x1, x2, x5 are important
eta <- 1.5*X$x1 - 2*X$x2 + 1*X$x5
prob <- 1 / (1 + exp(-eta))
y <- rbinom(n, 1, prob)

# Step 1: Build multi-path forest
forest <- build_paths(
  X = X,
  y = y,
  family = "binomial",
  K = 8,
  eps = 1e-6,
  delta = 2,
  L = 50
)

# Step 2: Compute stability
stab <- stability(
  X = X,
  y = y,
  family = "binomial",
  B = 50,
  K = 8,
  eps = 1e-6,
  delta = 2,
  L = 50
)

# Step 3: Select plausible models
plaus <- plausible_models(
  forest = forest,
  pi = stab$pi,
  Delta = 2,
  tau = 0.6
)

# View plausible models
print(plaus$plausible_models)
```

**Evaluate classification performance:**
```r
# Run complete pipeline
result <- multipath_aic(
  X = X,
  y = y,
  family = "binomial",
  K = 8,
  B = 50,
  Delta = 2,
  tau = 0.6
)

# Compute confusion matrix and metrics for best model
confusion_metrics(result, model_index = 1, cutoff = 0.5)
```

## Parameters

### Core Parameters

- **`K`**: Maximum number of forward selection steps (default: `min(ncol(X), 10)`)
- **`eps`**: Minimum AIC improvement threshold (default: `1e-6`)
- **`delta`**: AIC tolerance for keeping near-best models at each step (default: `2`)
- **`L`**: Maximum models retained per step (default: `100`)

### Stability Parameters

- **`B`**: Number of bootstrap resamples (default: `100`)
- **`resample_fraction`**: Fraction of data per resample (default: `0.8`)

### Plausibility Parameters

- **`Delta`**: AIC tolerance for plausible model set (default: `2`)  
  *Justification: Models within 2 AIC units are considered statistically equivalent (Burnham & Anderson, 2002)*
- **`tau`**: Minimum average stability threshold (default: `0.6`)  
  *Justification: Retains models with variables appearing in >60% of resamples, indicating robust selection*

## Real Data Example

The package includes a detailed vignette using the diabetes progression dataset:
```r
# View the vignette
vignette("diabetes-analysis", package = "multipathaic")
```

## Key Functions

| Function | Description |
|----------|-------------|
| `build_paths()` | Multi-path forward selection with AIC branching |
| `stability()` | Bootstrap-based variable stability estimation |
| `plausible_models()` | Filter models by AIC + stability |
| `multipath_aic()` | Complete pipeline (Algorithms 1-3) |
| `confusion_metrics()` | Classification performance metrics |

## Citation

If you use this package, please cite:
```
Obuobi, M., Jiang, J., & Rahmati, F. (2025). multipathaic: Multi-Path Stepwise 
Selection with AIC. R package version 0.1.0. 
https://github.com/R-4-Data-Science/FinalProjectmultipathaic
```

## References

- Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach* (2nd ed.). Springer.
- Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression. *The Annals of Statistics*, 32(2), 407-499.

## Authors

- Michael Obuobi (Auburn University)
- Jinchen Jiang (Auburn University)
- Far Rahmati (Auburn University)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Issues and Contributions

Report bugs or request features at: https://github.com/R-4-Data-Science/FinalProjectmultipathaic/issues
