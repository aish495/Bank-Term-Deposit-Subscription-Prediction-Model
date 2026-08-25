# Bank Term Deposit Subscription Prediction

Predicting whether a bank customer will subscribe to a term deposit, using Logistic Regression.

## Business Problem

A bank runs marketing campaigns (calls, emails, messages) to sell term deposits. Contacting every client is costly, and historically only a small fraction subscribe. This project builds a classification model to predict whether a client will subscribe (`y = yes/no`) based on client and campaign attributes, so the bank can target customers more efficiently.

## Dataset

`bank_dataset.csv` — client and campaign data including demographics (age, job, marital status, education), financial info (loans, defaults), contact details (month, day, communication type), campaign history, and macroeconomic indicators (employment rate, consumer price index, euribor rate, etc.).

## Workflow

1. Load & understand the data
2. Data cleaning & preprocessing (drop duplicates, handle `duration` leakage, encode `pdays`)
3. Exploratory Data Analysis (EDA)
4. Feature engineering & encoding (one-hot encoding, scaling)
5. Train/test split
6. Train Logistic Regression model
7. Evaluate performance (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)

## Key Notes

- **`duration` column dropped** — it's only known after a call is made, so keeping it would leak future information into a model meant to predict *before* the call.
- **Class imbalance** — only ~11% of clients subscribe, so accuracy alone is misleading. Precision, recall, F1, and ROC-AUC are used instead, along with `class_weight="balanced"`.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (balanced) | 0.834 | 0.342 | 0.567 | 0.427 | 0.762 |
| Logistic Regression (plain) | 0.902 | 0.696 | 0.178 | 0.283 | 0.772 |

The balanced model catches more actual subscribers (higher recall) at the cost of more false positives — useful if the goal is to identify as many potential subscribers as possible for outreach.

## Requirements

- Python 3
- pandas, numpy, matplotlib, seaborn, scikit-learn,

## How to Run

1. Place `bank_dataset.csv` in the same folder as the notebook.
2. Open `Bank Term Deposit Subscription Prediction`.ipynb
` and run all cells top to bottom.
