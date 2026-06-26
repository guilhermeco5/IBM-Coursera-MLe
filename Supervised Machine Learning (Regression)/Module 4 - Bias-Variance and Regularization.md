# Module 4 — Bias-Variance Trade-off and Regularization

> Supervised Machine Learning (Regression) — IBM / Coursera

## Bias-Variance Trade-off and Regularization (Ridge, LASSO, Elastic Net)

### Bias-Variance Trade-off

Choosing the level of complexity:

- **Polynomial degree = 1** → poor at both training and predicting, too simple.
- **Polynomial degree = 4** → just right for both training and predicting.
- **Polynomial degree = 15** → good at training and poor at predicting.

**Bias and Variance: intuition**

- **Bias** — a tendency to miss.
- **Variance** — a tendency to be inconsistent.

Ideally, we want outcomes that are highly consistent predictions and close to perfect on average (low variance and low bias). Here, *tendency* means the expectation of out-of-sample behavior over many training-set samples.

**3 sources of model error:**

- **Bias** → being wrong. Tendency of predictions to miss true values.
  - Worsened by missing information and overly simplistic assumptions.
  - Misses the real pattern (underfit).
- **Variance** → being unstable. Tendency of predictions to fluctuate.
  - Characterized by sensitivity of output to small changes in input data.
  - Often due to an overly complex or poorly-fit model.
- **Irreducible Error** → unavoidable randomness. Intrinsic uncertainty/randomness.
  - Present in even the best possible model.

**Bias-Variance trade-off, summarized:**

- Model adjustments that decrease bias often increase variance, and vice versa.
- The bias-variance trade-off is analogous to a complexity trade-off.
- Finding the best model means choosing the right level of complexity.
- We want a model elaborate enough to not underfit, but not so exceedingly elaborate that it overfits.

**Discussion:**

- The higher the degree of a polynomial regression, the more complex the model (lower bias, higher variance).
- At lower degrees, we see visual signs of bias: predictions are too rigid to capture the curve pattern in the data.
- At higher degrees, we see visual signs of variance: predictions fluctuate wildly because of the model's sensitivity.
- The goal is to find the right degree, so that the model has sufficient complexity to describe the data without overfitting.

**Example:**

- **Polynomial degree = 1** → high bias and low variance.
- **Polynomial degree = 4** → just right.
- **Polynomial degree = 15** → low bias, high variance.

### Regularization and Model Selection

**Tuning the model** — can we tune with more granularity than choosing polynomial degrees? Yes, by using regularization.

**What does regularization accomplish?**

$$M(W) + \lambda R(W)$$

- $M(W)$ — model error
- $R(W)$ — function of the estimated parameter(s)
- $\lambda$ — regularization strength parameter

→ Regularization adds an (adjustable) regularization strength parameter directly into the cost function. The parameter $\lambda$ (lambda) allows us to manage the complexity trade-off:

- More regularization introduces a simpler model, or more bias.
- Less regularization makes the model more complex and increases variance.
- If our model is overfit (variance too high), regularization can improve the generalization error and reduce variance.

→ This $\lambda$ adds a penalty proportional to the size of the estimated model parameter (or a function of the parameter).
→ Increasing the cost-function penalty controls the amount of regularization.

### Regularization and Feature Selection

- → Regularization performs feature selection by shrinking the contribution of features.
- → For L1 regularization, this is accomplished by driving some coefficients to zero.
- → Feature selection can also be performed by removing features.

**Why is feature selection important?**

- → Reducing the number of features can prevent overfitting.
- → For some models, fewer features can improve fitting time and/or results.
- → Identifying the most critical features can improve model interpretability.

### Ridge Regression

A regularization technique in linear regression to prevent overfitting by penalizing large coefficients.

**Ridge regression cost function and penalty:**

- → Ridge regression modifies the original linear regression cost function by adding a penalty term proportional to the *square* of the coefficients, multiplied by a regularization parameter $\lambda$.
- → This penalty discourages large coefficient values, balancing the goal of minimizing prediction error with controlling model complexity.

**Importance of feature scaling:**

- → Because the penalty depends on coefficient magnitude, features must be standardized (mean zero, unit variance) to ensure fair penalization across variables with different scales.
- → Without scaling, variables with larger scales could dominate the penalty and distort the model.

**Effect of $\lambda$ on model complexity:**

- → Increasing $\lambda$ increases the penalty on coefficients, shrinking them towards zero (but not exactly zero), which reduces model variance but introduces bias.
- → Cross-validation is used to select the optimal $\lambda$ that balances bias and variance for best predictive performance.
- → Higher $\lambda$ values lead to smaller coefficients and simpler models, while $\lambda = 0$ corresponds to no regularization and potentially overfitting.

**Summary of Ridge regression:** it helps control model complexity by penalizing large coefficients, requires feature scaling, and tunes $\lambda$ to achieve a good bias-variance trade-off. The next topic covers another regularization method: LASSO regression.

### LASSO Regression

A regularization technique for linear regression, compared with Ridge regression and introducing Elastic Net as a hybrid approach.

- → LASSO uses the absolute value (L1 norm) of coefficients for penalization, unlike Ridge, which uses the squared coefficients (L2 norm).
- → It tends to shrink some coefficients to zero, effectively performing feature selection and reducing model complexity.

**Comparison with Ridge regression:**

- → LASSO is less influenced by outliers due to the absolute-value penalty, and can eliminate less important features.
- → Ridge regression shrinks all coefficients more uniformly and is computationally faster to converge.
- → Choosing between them depends on goals like prediction accuracy, interpretability, and computational efficiency.

### Elastic Net Regularization

- → Elastic Net combines LASSO and Ridge penalties, controlled by parameters $\lambda$ and $\alpha$.
- → It balances the benefits of both methods, allowing tuning between feature selection and coefficient shrinkage.
- → Optimal hyperparameters are found via cross-validation to avoid underfitting or overfitting.
