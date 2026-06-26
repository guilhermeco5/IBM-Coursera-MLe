# Módulo 2 — Data Splits and Polynomial Regression

> Supervised Machine Learning (Regression) — IBM / Coursera

## Data Splits and Polynomial Regression

### Training and Test Splits

Using training and test data:

- **Training Data** → Fit the model
- **Test Data** → Measure performance (predict label, compare with actual value, measure error)

### Fitting Training and Test Data

```
Training (X_train, Y_train) → model(X_train, Y_train).fit() → model

Test Data (X_test) → model.predict(X_test) → Y_predict
                  → Y_test (actuals) → error_metric(Y_test, Y_predict) → test error
```

### Train-Test Split: The Syntax

Import the train/test split function:

```python
from sklearn.model_selection import train_test_split
```

Split the data and put 30% into the test set:

```python
train, test = train_test_split(data, test_size=0.3)
```

Other method for splitting data:

```python
from sklearn.model_selection import ShuffleSplit
```

### Polynomial Regression

→ Addition of polynomial features.

Capture higher-order features of data by adding polynomial features.

"Linear regression" means **linear combinations of features**.

→ Can also include variable interactions.

How is the correct functional form chosen? Check the relationship of each variable with the outcome.

### Enhancing the Linear Model

Adjusting the standard linear approach to regression by adding polynomial features is one of many approaches to dealing with the two fundamental problems:

- Prediction
- Interpretation

As we move into model evaluation, keep in mind that the same tools are useful for evaluating a wide variety of regression and classification problems.

### Extending the Linear Model

In addition to polynomial features, we will also examine several additional variants of the standard model, using many for both regression and classification. Some examples include:

- Logistic Regression
- K-Nearest Neighbors
- Decision Trees
- Support Vector Machines
- Random Forests
- Ensemble Methods
- Deep Learning approaches

### Polynomial Features: The Syntax

Import the class containing the transformation method:

```python
from sklearn.preprocessing import PolynomialFeatures
```

Create an instance of the class:

```python
polyFeat = PolynomialFeatures(degree=2)
```

Create the polynomial features and then transform the data:

```python
polyFeat = polyFeat.fit(X_data)
X_poly = polyFeat.transform(X_data)
```
