# Predicting Youth Alcohol Use
### A Machine Learning Approach Using Decision Trees, Random Forest, Bagging, and Boosting

---

## Overview

This project applies machine learning techniques to investigate factors associated with youth alcohol use using data from the **National Survey on Drug Use and Health (NSDUH)**. Three predictive models were developed to capture different dimensions of alcohol use:

- **Binary classification** — whether a youth used alcohol in the past year
- **Multiclass classification** — frequency of alcohol use among drinkers
- **Regression** — age of first alcohol use

The findings aim to surface actionable insights for prevention programs, school administrators, and policymakers by identifying the behavioral, attitudinal, and socioeconomic factors most associated with youth alcohol use.

---

## Data Source

**National Survey on Drug Use and Health (NSDUH)** — administered by the Substance Abuse and Mental Health Services Administration (SAMHSA)

- Population: general civilian population aged 12 and older in the United States
- Topics covered: lifetime, past-year, and past-month substance use; age at first use; treatment history; substance use disorders; perceived risk; protective factors
- Project scope: narrowed to **23 variables** most relevant to youth alcohol use

---

## Project Structure

```
├── data/
│   ├── df                        # Full cleaned dataset (~9,000 rows)
│   └── dfdr                      # Drinkers-only subset (~2,200 rows)
├── notebooks/
│   ├── 01_preprocessing.ipynb    # Variable selection, encoding, imputation, target engineering
│   ├── 02_eda.ipynb              # Exploratory data analysis, class imbalance plots
│   ├── 03_binary.ipynb           # Binary classification — Decision Tree
│   ├── 04_multiclass.ipynb       # Multiclass classification — Bagging
│   └── 05_regression.ipynb       # Regression — Gradient Boosting
├── README.md
└── requirements.txt
```

---

## Preprocessing

- Dataset narrowed to **23 variables**: 3 numeric, 2 ordinal categorical, remainder binary categorical
- Redundant variables (same question in different formats) removed
- Binary variables encoded as **0/1** for model compatibility
- Missing rows dropped (~10% of dataset) after IterativeImputer was deemed potentially noise-introducing
- **Three target variables** engineered from source column:
  - `User` — binary (0 = no use, 1 = any use in past year)
  - `Frequency` — multiclass (0–3, four balanced frequency groups among drinkers)
  - `Age of First Use` — continuous regression target

---

## Models & Results

### Binary Classification — Decision Tree

| Parameter | Value |
|---|---|
| Algorithm | Decision Tree |
| criterion | gini |
| max_depth | 7 |
| min_samples_split | 20 |
| min_samples_leaf | 1 |
| class_weight | balanced |

| Metric | Value |
|---|---|
| Training Accuracy | 66.64% |
| Test Accuracy | 64.29% |
| Train/Test Difference | 2.35% |
| Macro F1 | 0.58 |
| Weighted F1 | 0.68 |
| Most Important Feature | TV usage limit |

---

### Multiclass Classification — Decision Tree + Bagging

| Parameter | Value |
|---|---|
| Algorithm | BaggingClassifier (DecisionTree base) |
| criterion | gini |
| max_depth | 10 |
| min_samples_split | 2 |
| min_samples_leaf | 5 |
| n_estimators | 15 |
| max_samples | 1.0 |
| bootstrap | True |

| Metric | Value |
|---|---|
| Training Accuracy | 61.53% |
| Test Accuracy | 32.14% |
| Train/Test Difference | 29.39% |
| Macro F1 | 0.30 |
| Weighted F1 | 0.31 |
| Most Important Feature | Days of missed school |

---

### Regression — Gradient Boosting

| Parameter | Value |
|---|---|
| Algorithm | GradientBoostingRegressor |
| criterion | squared_error |
| max_depth | 3 |
| min_samples_split | 5 |
| min_samples_leaf | 1 |
| n_estimators | 30 |
| learning_rate | 0.1 |

| Metric | Value |
|---|---|
| Train MSE | 5.392 |
| Test MSE | 5.417 |
| Train RMSE | 2.322 |
| Test RMSE | 2.327 |
| Mean CV RMSE | 2.410 |
| CV Std | 0.062 |
| Most Important Feature | Frequency of fighting at school |

---

## Key Findings

Across all three models, a consistent set of themes emerged as the most influential predictors of youth alcohol use:

- **Parental involvement** — TV usage limits, nighttime monitoring, parental praise, and conversations about alcohol dangers consistently appeared as protective factors
- **School disengagement** — days of missed school was a top predictor in both multiclass and regression models; absenteeism is a co-occurring risk marker
- **Delinquent behavior** — frequency of stealing appeared in both binary and regression models, suggesting alcohol use occurs as part of broader risk-taking patterns
- **Peer attitudes** — friends' attitudes toward drinking was influential in the binary model, confirming the role of social norms in initiation
- **Socioeconomic factors** — poverty level, total family income, and government assistance appeared across all three models

---

## Limitations

- **Severe class imbalance** — 92% non-users in original dataset forced significant narrowing of scope
- **Scope restriction** — multiclass and regression limited to drinkers only (~2,200 rows), constraining ensemble learning power
- **Artificial class boundaries** — frequency classes constructed for balance rather than natural data separability
- **Binary feature granularity** — most predictors are 0/1 encoded, lacking resolution to distinguish fine-grained frequency differences
- **Cross-sectional data** — NSDUH captures a single point in time, limiting causal inference
- **Elbow plot instability** — no optimal number of trees identified in multiclass or regression, indicating weak signal in data

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
imbalanced-learn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Real-World Applications

- **Targeted prevention programs** — feature importance findings identify which protective factors to prioritize
- **Early identification** — binary model can help flag youth at higher risk of alcohol use
- **Resource allocation** — policymakers can direct resources toward families, schools, and communities with the highest risk profiles
- **School-based intervention** — consistent appearance of absenteeism across models supports school attendance as an intervention target

---

## Acknowledgements

Data sourced from the **National Survey on Drug Use and Health (NSDUH)**, administered by the Substance Abuse and Mental Health Services Administration (SAMHSA). This project was completed as part of a graduate-level machine learning course.
