# bank-churn-prediction
Bank Customer Churn Prediction  |  Classification Modelling Project
# Bank Customer Churn Prediction — Why 80% Accuracy is a Failing Grade

> Classification · Imbalanced Data · Python · NumPy from Scratch

---

## The Problem

Customer churn costs banks 5× more than retention. But predicting churn has a hidden trap: in a dataset where only 20% of customers churn, a model that predicts "no churn" for everyone achieves **80% accuracy — and is completely useless**.

This project builds a full ML pipeline from scratch in NumPy (no scikit-learn model classes) to navigate this trap and produce a model that genuinely identifies at-risk customers.

---

## Key Results

| Model | Accuracy | Churn Recall | AUC | Verdict |
|---|---|---|---|---|
| Logistic Regression (baseline) | 80% | **0%** | 0.63 | Useless |
| ★ LR + SMOTE + Class Weights | 67% | **63%** | **0.71** | Best overall |
| SVM (no preprocessing) | 0% | — | 0.50 | Random guessing |
| SVM (K-Fold + Standardisation) | 75% | ~46% | 0.65 | Competitive |

The model with **lower accuracy (67%) is the better model.** It correctly flags 247 real churners that the "accurate" model misses entirely.

---

## The Central Insight

> *"Accuracy measures how often you're right. Recall measures whether you're right about the things that matter."*

In banking, each correctly identified churner represents a retention opportunity worth thousands in lifetime value. A false positive (a retained customer who wouldn't have left anyway) costs only a small incentive offer. **The business cost of the two error types is not symmetric — and your metric choice should reflect that.**

---

## Approach

**1. Data & EDA (10,000 customers, 11 features)**
- Age (r=+0.29) and active member status (r=−0.16) are strongest predictors
- Germany churns at ~32% vs ~16% for France/Spain
- Female customers churn at 25.1% vs 16.5% for males (Chi-square p ≈ 2.24×10⁻²⁶)
- Age capped at 70 (capping method), two interaction features engineered

**2. Handling 80/20 Imbalance**
- SMOTE: synthesised minority class samples (7,963 → 7,963 per class)
- Custom class weights: penalised misclassification of churners during gradient descent
- Undersampling: tested as alternative baseline

**3. Models — All Built from Scratch in NumPy**
- **Logistic Regression:** sigmoid function, binary cross-entropy loss, gradient descent (1,000 epochs, lr=0.01)
- **SVM:** hinge loss, gradient descent with margin constraints, 5-fold cross-validation
- **Random Forest:** bootstrap sampling, Gini impurity, majority voting — computationally infeasible on 10,000 rows without optimised libraries (honest limitation, documented)

---

## Tech Stack

| Tool | Usage |
|---|---|
| Python | Core implementation |
| NumPy | All models built from scratch |
| Pandas | Data manipulation and EDA |
| Matplotlib / Seaborn | Visualisation |
| imbalanced-learn | SMOTE implementation |
| Scikit-Learn | Metrics only (confusion matrix, AUC, K-Fold) |

---

## Repository Structure

```
bank-churn-prediction/
├── bank_churn.ipynb      # Main notebook
├── Bank.csv              # Dataset (if permitted to share)
├── README.md
└── LICENSE
```

---

## How to Run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn

jupyter notebook bank_churn.ipynb
```

---

## Author

**Shyamali Naik** · MSc Data Analytics · Queen Mary University of London · 2025

[LinkedIn](https://www.linkedin.com/in/shyamali-naik-91n78) · [Portfolio](https://shyamalinaik369-ops.github.io)

---

*© 2025 Shyamali Naik. Licensed under MIT.*
