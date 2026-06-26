# Module 1 — Intro to Supervised ML and Linear Regression

> Supervised Machine Learning (Regression) — IBM / Coursera

## Introduction to Supervised Machine Learning and Linear Regression

In this module, you'll identify the three primary types of machine learning — Supervised, Unsupervised, and Reinforcement. You'll explore real-world applications across various industries and distinguish between prediction and interpretation tasks. You'll examine the fundamentals of linear regression, including key assumptions and use cases, and learn how to measure model error, improve predictive accuracy, and use practical tools to train and evaluate linear regression models.

### Machine Learning in our daily lives

- Spam filtering
- Web search
- Postal mail routing
- Fraud detection
- Movie recommendations
- Vehicle driver assistance
- Web advertisements
- Social networks
- Speech recognition

### Fit Parameters vs Hyperparameters

ML framework (applies to Supervised Machine Learning models):

$$Y_p = f(\Omega, x)$$

Our framework estimates a relationship between the features and the target:

- Here, $\Omega$ (the **Fit Parameters**) involves aspects of the model we estimate (fit) using the data.
- To implement our approach, we make decisions regarding how to produce these estimates.
- These decisions lead to **Hyperparameters**, which are an important part of the machine learning workflow (though not explicit components of the model).

### Machine Learning Framework

ML framework (applies to every Supervised Machine Learning model):

$$Y_p = f(\Omega, x)$$

Two main modeling approaches:

- **Regression** — $Y$ is numeric. E.g.: stock price, box office revenue, location (x, y coordinates).
- **Classification** — $Y$ is categorical. E.g.: face recognition, customer churn, which word comes next.

**How does Machine Learning work?**

$$Y_p = f(\Omega, x)$$

- $x$ — input
- $Y_p$ — output (values predicted by the model)
- $f(\cdot)$ — prediction function that generates predictions from $x$ and $\Omega$
- $j(y, y_p)$ — loss

Most ML models define a quantitative score for how good our predictions are; it typically measures how close our predictions are to the true values.

**Update rule:** using features $x$ and outcome $y$, choose parameters $\Omega$ to minimize loss.
