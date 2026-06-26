# Module 3 — EDA, Visualization and Feature Engineering

> Exploratory Data Analysis for Machine Learning — IBM / Coursera

## What is Exploratory Data Analysis? (EDA)

An approach to analyzing datasets to summarize their main characteristics, often with visual methods.

### Why is EDA useful?

- Allows us to get an initial feel for the data.
- Lets us determine if the data makes sense, or if further cleaning or more data is needed.
- Helps to identify patterns and trends in the data (these can be just as important as findings from modeling).

### Techniques for EDA

- **Summary Statistics** — Average, Median, Min, Max, Correlations, etc.
- **Visualizations** — Histograms, Scatter Plots, Box plots, etc.

### Tools for EDA

- **Data Wrangling** — Pandas
- **Visualization** — Matplotlib, Seaborn

### EDA: Job Applicant Summary Statistics

Suppose we want to examine characteristics of job applicants:

- **Average** — look at the average of all interview scores (perhaps by city or job function).
- **Max** — look at the most common words applicants use in application materials.
- **Correlations** — look at the correlations between technical assessments and years of experience (perhaps by type of experience).

## Sampling from DataFrames

```python
# Sample 5 rows without replacement
sample = data.sample(n=5, replace=False)
print(sample.iloc[:, -3:])
```

There are many reasons to consider random samples from DataFrames:

- For large data, a random sample can make computation easier.
- We may want to train models on a random sample of the data.
- We may want to over- or under-sample observations when outcomes are uneven.

## EDA with Visualization Libraries

- **Matplotlib**
- **Pandas**
- **Seaborn**
  - Statistically-focused plotting methods
  - Global preferences incorporated from Matplotlib

**Pandas DataFrame approach:**

```python
import matplotlib.pyplot as plt
plt.plot(data.sepal_length,
         data.sepal_width,
         ls='', marker='o')
```

**Scatter plots with multiple layers:**

```python
# First plot statement
plt.plot(data.sepal_length,
         data.sepal_width,
         ls='', marker='o',
         label='sepal')

# Second plot statement
plt.plot(data.petal_length,
         data.petal_width,
         ls='', marker='o',
         label='petal')
```

**Histograms:**

```python
plt.hist(data.sepal_length, bins=25)
```

**Customizing plots:**

```python
# Matplotlib syntax
fig, ax = plt.subplots()
ax.barh(np.arange(10),
        data.sepal_width.iloc[:10])

# Set position of ticks and tick labels
ax.set_yticks(np.arange(0.4, 10.4, 1.0))
ax.set_yticklabels(np.arange(1, 11))
ax.set(xlabel='xlabel', ylabel='ylabel', title='Title')
```

### Summary

- **Matplotlib** — the primary library for creating plots and graphs in Python, offering flexibility and numerous features. Pandas provides a convenient wrapper around Matplotlib, allowing for easier plotting, though it is less powerful than using Matplotlib directly.
- **Seaborn** — built on top of Matplotlib, it simplifies the creation of aesthetically pleasing and statistically interesting plots. It integrates with Matplotlib, allowing for consistent styling when switching between the two libraries.
- **Plotting techniques:**
  - Basic scatter plots can be created using Matplotlib, with options to differentiate data points by color and add legends.
  - Histograms can be plotted to visualize the distribution of a single variable.
  - Object-oriented plotting with Matplotlib allows for more customization, such as setting specific tick labels and titles.
  - Using Pandas for plotting can simplify the process, especially when grouping data by categories.

## Customizing Plots: by Group

```python
# Pandas DataFrame approach
data.groupby('species').mean() \
    .plot(color=['red', 'blue', 'black', 'green'],
          fontsize=10.0, figsize=(4, 4))
```

## Pair Plots for Features

```python
# Seaborn plot, feature correlations
sns.pairplot(data, hue='species', size=3)

# Seaborn hexbin plot
sns.jointplot(x=data['sepal_length'],
              y=data['sepal_width'],
              kind='hex')

# Seaborn plot, Facet Grid — first plot statement
plot = sns.FacetGrid(data,
                     col='species',
                     margin_titles=True)
plot.map(plt.hist, 'sepal_width', color='green')

# Second plot statement
plot = sns.FacetGrid(data,
                     col='species',
                     margin_titles=True)
plot.map(plt.hist, 'sepal_length', color='blue')
```

## Feature Engineering and Variable Transformation

**Background:** Models used in Machine Learning workflows often make assumptions about the data. A common example is the linear regression model, which assumes a linear relationship between observations and target (outcome) variables.

### Transformation of Data Distributions

Predictions from linear regression models assume residuals are normally distributed. Features and predicted data are often skewed (distorted away from the center). Data transformation can solve this issue.

```python
# Useful transformation functions
from numpy import log, log1p
from scipy.stats import boxcox
```

- **Log transformations** can be useful for linear regression, since linear regression involves combinations of features.
- **Polynomial features** — we can estimate high-order relationships by adding polynomial features. This allows us to use the same "linear" model even with higher-order polynomials.

```python
# Import the class containing the transformation method
from sklearn.preprocessing import PolynomialFeatures

# Create an instance of the class (choose number of degrees)
polyFeat = PolynomialFeatures(degree=2)

# Create the polynomial features and then transform the data
polyFeat = polyFeat.fit(X_data)
x_poly = polyFeat.transform(X_data)
```

## Feature Encoding

### Variable Selection: Background

Involves choosing the set of features to include in the model. Variables must often be transformed before they can be included. In addition to log and polynomial transformations, this can involve:

- **Encoding** — converting non-numeric features to numeric features
- **Scaling** — converting the scale of numeric data so they are comparable

The appropriate method of scaling or encoding depends on the type of feature.

### Feature Encoding: types of features

Encoding is often applied to categorical features that take non-numeric values. Two primary types:

- **Nominal** — categorical variables take values in unordered categories (e.g. Red, Blue, True, False)
- **Ordinal** — categorical variables take values in ordered categories (e.g. high, medium, low)

### Feature Encoding: Approaches

- **Binary encoding** — converts variables to either 0 or 1; suitable for variables that take two possible values.
- **One-hot encoding** — converts variables that take multiple values into binary (0, 1) variables, one for each category. This creates several new variables.
- **Ordinal encoding** — converts ordered categories to numerical values, usually by creating one variable that takes an integer equal to the number of categories.

### Feature Scaling: Approaches

- **Standard scaling** — converts features to standard normal variables (by subtracting the mean and dividing by the standard error).
- **Min-max scaling** — converts variables to continuous variables in the $(0, 1)$ interval by mapping minimum values to 0 and maximum values to 1. This type of scaling is sensitive to outliers.
- **Robust scaling** — similar to min-max scaling, but instead maps the interquartile range (the 75th percentile value minus the 25th percentile value) to $(0, 1)$. This means the variable itself can take values outside of the $(0, 1)$ interval.

## Common Variable Transformations

```python
# Useful functions for scaling variables
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Useful functions for encoding categorical variables
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from pandas import get_dummies

# Useful functions for encoding ordinal variables
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OrdinalEncoder
```

| Feature type | | Transformation |
|---|---|---|
| **Continuous** → numerical values | | Standard, Min-Max, Robust scaling |
| **Nominal** → unordered categorical | | Binary, one-hot encoding (0, 1) |
| **Ordinal** → ordered categorical | | Ordinal encoding (0, 1, 2, 3) |
