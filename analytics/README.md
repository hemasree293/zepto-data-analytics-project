# Module 2 — Titanic Data Analytics and Machine Learning

## Overview

This module performs exploratory data analysis (EDA), data cleaning, classification modeling, class imbalance analysis, hyperparameter tuning, and regression analysis using the Titanic dataset.

The raw Titanic dataset was loaded once using Seaborn and saved as `titanic.csv` for offline use. All subsequent analysis and modeling uses the cleaned dataset.

---

## 1. Data Loading and Initial Exploration

The Titanic dataset was loaded using:

```python
sns.load_dataset("titanic")

## Final Model Recommendation

The three classification models produced different performance profiles across accuracy, precision, recall, F1, and ROC-AUC. Logistic Regression achieved an accuracy of 0.8090 and the highest ROC-AUC of 0.8610, while Random Forest achieved the highest F1 score of 0.7424 and recall of 0.7206 among the three models. The tuned Random Forest achieved a cross-validation F1 of 0.7408 with an OOB score of 0.8214. Based on these measured results, Logistic Regression provides stronger ROC-AUC performance, while Random Forest provides stronger F1 and recall, so the appropriate model depends on whether ranking/discrimination or positive-class identification is the primary evaluation objective.