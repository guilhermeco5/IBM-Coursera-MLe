# Module 3 — Cross Validation

> Supervised Machine Learning (Regression) — IBM / Coursera

## Cross Validation

Also good to use for model selection. It helps to identify the issues in our models.

### Model Complexity vs Error

**Underfitting** — training and cross-validation error are both high.

### Cross Validation Approaches

- **K-fold Cross Validation** — using each of $K$ subsamples as a test sample.
- **Leave-one-out Cross Validation** — using each observation as a test sample.
- **Stratified Cross Validation** — K-fold cross validation with representative samples.

### Cross Validation: The Syntax

Import the cross-validation function:

```python
from sklearn.model_selection import cross_val_score
```

Perform cross-validation with a given model:

```python
cross_val = cross_val_score(model, X_data, Y_data, cv=4,
                            scoring='neg_mean_squared_error')
```

Other methods for cross validation:

```python
from sklearn.model_selection import KFold, StratifiedKFold
```
