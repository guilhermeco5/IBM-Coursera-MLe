# Module 4 — Inferential Statistics and Hypothesis Testing

> Exploratory Data Analysis for Machine Learning — IBM / Coursera

## Inferential Statistics and Hypothesis Testing

### Introduction

- **Estimation** — the application of an algorithm, for example taking an average.
- **Inference** — putting an accuracy on the estimate (e.g. the standard error of an average).

### Machine Learning and statistical inference are similar

(A case of computer science borrowing from a long history in statistics.) In both cases, we're using data to learn/infer qualities of a distribution that generated the data (often termed the *data-generating process*).

We may care either about the whole distribution or just features (e.g. the mean). Machine Learning applications that focus on understanding parameters and individual effects involve more tools from statistical inference (some applications are focused only on results).

### Example: Customer Churn

Customer churn occurs when a customer leaves a company. Data related to churn may include a target variable for whether or not the customer left. Features could include:

- the length of time as a customer
- the type and amount purchased
- other customer characteristics

Churn prediction is often approached by predicting a score for individuals that estimates the probability the customer will leave.

**Customer Churn: Estimation** — estimation of factors driving customer churn involves measuring the impact of each factor in predicting churn. Inference involves determining whether these measured impacts are statistically significant.

Example dataset — IBM Cognos Customer Churn Dataset:

```python
# Seaborn hexbin plot
sns.jointplot(x=df[labels['months']],
              y=df[labels['monthly']],
              kind='hex')
```

## Parametric vs Non-Parametric

If inference is about trying to find out the Data-Generating Process (DGP), then we can say that a statistical model of the data is a set of possible distributions (or maybe even regressions).

A **parametric model** is a particular type of statistical model: it is also a set of distributions or regressions, but they have a finite number of parameters.

### Non-Parametric Statistics

In non-parametric statistics, we make fewer assumptions. In particular, we don't assume that the data belong to any particular distribution (also called distribution-free inference). This doesn't mean that we know nothing, though!

### Non-Parametric Inference

An example of non-parametric inference is creating a distribution of the data (CDF, or cumulative distribution function) using a histogram. In this case, we are not specifying parameters.

### Parametric Models

A parametric model is a particular type of statistical model: it is also a set of distributions or regressions, but they have a finite number of parameters. An example of a parametric model: the Normal distribution.

**Example: Customer Lifetime Value** — an estimate of the customer's value to the company. Data might include:

- The expected length of time as a customer
- The expected amount spent over time

To estimate lifetime value, we make assumptions about the data. These can be parametric (assuming a specific distribution) or non-parametric.

**Parametric Models: Maximum Likelihood** — the most common way of estimating in a parametric model is through maximum likelihood estimation (MLE). The likelihood function is related to probability and is a function of the parameters.

### Commonly Used Distributions

- Uniform
- Gaussian / Normal (Central Limit Theorem)
- Log-Normal
- Exponential
- Poisson

### Frequentist vs Bayesian Statistics

A **Frequentist** is concerned with repeated observations in the limit. Processes may have true frequencies, but we are interested in modeling probabilities as many, many repeats of an experiment.

Frequentist approach:

1. Derive the probabilistic property of a procedure.
2. Apply the probability directly to the observed data.

A **Bayesian** describes parameters by probability distributions. Before seeing any data, a *prior* distribution (based on the experimenter's belief) is formulated. This prior distribution is then updated after seeing data (a sample from the distribution). After updating, the distribution is called the *posterior* distribution.

We use much of the same math and the same formulas in both Frequentist and Bayesian statistics — the element that differs is the interpretation.

## Hypothesis Testing

### Introduction

A hypothesis is a statement about a population parameter. We create two hypotheses:

- The null hypothesis $H_0$
- The alternative hypothesis $H_1$ (or $H_A$)

We decide which one to call the null depending on how the problem is set up.

### Decision Rules

A hypothesis-testing procedure gives us a rule to decide:

- For which values of the test statistic we accept $H_0$
- For which values of the test statistic we reject $H_0$ and accept $H_1$

You may hear some people say that you can reject $H_0$ but that you never accept $H_1$. Here, this doesn't matter very much, since we are using hypothesis testing in order to decide which of two paths to take in the project.

### Bayesian Approach

In the Bayesian interpretation, we don't get a decision boundary. Instead, we get updated (posterior) probabilities. We need priors for each hypothesis. In this case, we randomly chose the coin flip:

$$P(H_1) = P(H_2) = \tfrac{1}{2}$$

Since we have no way, before seeing the data, to determine the coin that was chosen, we just assign $\tfrac{1}{2}$ to each.

Updating priors after seeing the data (3 heads), via Bayes' Rule:

$$P(H_1 \mid x) = \frac{P(x \mid H_1)\, P(H_1)}{P(x)}$$

We can write out the ratio:

$$\frac{P(H_1 \mid x)}{P(H_2 \mid x)} = \frac{P(H_1)\, P(x \mid H_1)}{P(H_2)\, P(x \mid H_2)}$$

The priors are multiplied by the likelihood ratio, which does not depend on the priors. The likelihood ratio tells us how we should update the priors in reaction to seeing a given set of data.

### Neyman-Pearson Interpretation

The Neyman-Pearson paradigm (1933) is non-Bayesian. It gives an up-or-down vote on $H_0$ vs $H_1$.

**Example: Customer Churn** — churn occurs when a customer leaves a company. Data may include a target variable for whether or not the customer left. Features could include the length of time as a customer, the type and amount purchased, and other customer characteristics. Churn prediction is often approached by predicting a score that estimates the probability the customer will leave.

**Type I vs Type II Error** — suppose we use data on customer characteristics to predict who will churn over the next year. In our data, customers who have been with the company longer are less likely to churn. This could be due to an underlying effect, or due to chance:

- A **Type I error** occurs when the effect is due to chance, but we find it to be significant in the model.
- A **Type II error** occurs when we ascribe the effect to chance, but the effect is non-coincidental.

### Terminology

The likelihood ratio is called a **test statistic**: we use it to decide whether to accept/reject $H_0$.

- **Rejection region** — the set of values of the test statistic that lead to rejection of $H_0$.
- **Acceptance region** — the set of values of the test statistic that lead to acceptance of $H_0$.
- **Null distribution** — the test statistic's distribution when the null is true.

### Examples of testing scenarios

- **Marketing Intervention** — for a new direct-mail marketing campaign to existing customers, the null hypothesis $H_0$ suggests the campaign does not impact purchasing; the alternative $H_1$ suggests it has an impact.
- **Website Layout** — for a proposed change to a website layout, we may test a null hypothesis $H_0$ that the change has no impact on traffic, looking for evidence to reject it in favor of $H_1$ (that there is an impact).
- **Product Quality** — testing whether a product meets a size threshold $S$. A product is produced in various factories with expected size $S$. To confirm the product size meets the standard within a margin of error, the company might:
  - Randomly sample from each production source.
  - Establish $H_0$ (product size is not significantly different from $S$).
  - And $H_1$ (there is a significant deviation in product size).
  - Test whether $H_0$ can be rejected in favor of $H_1$, based on the observed mean and standard deviation.

### Significance level and p-values

The significance level (**alpha**) is chosen before testing to decide when to reject the null hypothesis, balancing the risk of Type I and Type II errors. Lower alpha values mean stricter criteria for rejecting the null — important in high-stakes scenarios like medical testing.

We know the distribution of the null hypothesis. To get a rejection region, we calculate the test statistic. We choose, before testing the data, the level at which we will reject the null hypothesis.

**Important terminology:**

- **p-value** — the smallest significance level at which the null hypothesis would be rejected.
- **Confidence interval** — the values of the statistic for which we accept the null.

**Example: Coin Tossing** — suppose we saw three heads in 10 rolls:

- $P(\text{3 or fewer heads}) = P(0) + P(1) + P(2) + P(3) = 17\%$
- Under the null hypothesis, a value this extreme occurs 17% of the time.
- $H_0$: the coin is fair and $P(H) = 0.5$
- $H_1$: the coin is unfair and $P(H) < 0.5$

Testing the null hypothesis:

- We know $H_0$ is distributed $\text{binom}(10, 0.5)$.
- Choose a p-value cutoff, say 5%.
- Calculate the CDF of 3 heads from a $\text{binom}(10, 0.5)$.
- CDF = 17.1% (above our cutoff).
- This is > 5%, so we don't reject $H_0$.

### F-Statistic

$H_0$: the data can be modeled by setting all betas to zero. Reject the null if the p-value is small enough.

### Power and Sample Size

If you do many 5% significance tests looking for a significant result, the chances of making at least one Type I error increase.

**Bonferroni Correction** — says "choose $P$ so that the probability of making a Type I error (assuming no effect) is 5%." It allows the probability of a Type I error to be controlled, but at the cost of power: effects either need to be larger, or the tests need larger samples, to be detected. Best practice is to limit the number of comparisons done to a few well-motivated cases.
