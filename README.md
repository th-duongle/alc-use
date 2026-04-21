# Predicting Youth Alcohol Use
### A Machine Learning Approach Using Decision Trees, Random Forest, Bagging, and Boosting

---

## Overview

This project applies machine learning techniques to investigate factors associated with youth alcohol use using data from the **National Survey on Drug Use and Health (NSDUH)**. Three predictive models were developed to capture different dimensions of alcohol use:

- **Binary classification**: whether a youth used alcohol
- **Multiclass classification**: frequency of alcohol use among drinkers
- **Regression**: age of first alcohol use

---

## Data Source

**National Survey on Drug Use and Health (NSDUH)** — administered by the Substance Abuse and Mental Health Services Administration (SAMHSA)

- Population: general civilian population aged 12 and older in the United States
- Topics covered: lifetime, past-year, and past-month substance use; age at first use; treatment history; substance use disorders; perceived risk; protective factors
- Project scope: narrowed to **23 variables** most relevant to youth alcohol use

---

## Preprocessing

- Dataset narrowed to **23 variables**: 3 numeric, 2 ordinal categorical, remainder binary categorical
- Redundant variables (same question in different formats) removed
- Binary variables encoded as **0/1** for model compatibility
- Missing rows dropped (~10% of dataset) 
- **Three target variables** engineered from source column:
  - `User` — binary (0 = no use, 1 = any use)
  - `Frequency` — multiclass (0–3, four balanced frequency groups among drinkers)
  - `Age of First Use` — continuous regression target

---

## Key Findings

Across all three models, a consistent set of themes emerged as the most influential predictors of youth alcohol use:

- **Parental involvement** — TV usage limits, nighttime monitoring, parental praise, and conversations about alcohol dangers consistently appeared as protective factors
- **School disengagement** — days of missed school was a top predictor in both multiclass and regression models; absenteeism is a co-occurring risk marker
- **Delinquent behavior** — frequency of stealing appeared in both binary and regression models, suggesting alcohol use occurs as part of broader risk-taking patterns
- **Socioeconomic factors** — poverty level, total family income, and government assistance appeared across all three models

---

## Limitations

- **Severe class imbalance** — 92% non-users in original dataset forced significant narrowing of scope
- **Scope restriction** — multiclass and regression limited to drinkers only (~2,200 rows), constraining ensemble learning power
- **Artificial class boundaries** — frequency classes constructed for balance rather than natural data separability
- **Binary feature granularity** — most predictors are 0/1 encoded, lacking resolution to distinguish fine-grained frequency differences
- **Cross-sectional data** — NSDUH captures a single point in time, limiting causal inference
- **Elbow plot instability** — no optimal number of base models identified in multiclass or regression, indicating weak signal in data


---
## Real-World Applications

- **Targeted prevention programs** — feature importance findings identify which protective factors to prioritize
- **Early identification** — binary model can help flag youth at higher risk of alcohol use
- **Resource allocation** — policymakers can direct resources toward families, schools, and communities with the highest risk profiles
- **School-based intervention** — consistent appearance of absenteeism across models supports school attendance as an intervention target

---

## Acknowledgements

Data sourced from the **National Survey on Drug Use and Health (NSDUH)**, administered by the Substance Abuse and Mental Health Services Administration (SAMHSA). This project was completed as part of a graduate-level machine learning course.
