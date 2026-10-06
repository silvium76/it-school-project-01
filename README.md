# Project #1 — Train and compare classic ML algorithms: Telco Customer Churn

**Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 columns.
**What I predict:** `Churn` — whether a customer leaves the phone company (Yes / No). 26.5% of the customers churned.
**Notebook:** [`proiect_1_churn.ipynb`](proiect_1_churn.ipynb) — runs top to bottom in about 3 minutes (the CSV is in the same folder).

## Answers to the 6 questions

1. **Classification or regression? Which mistake costs more?** Binary classification (two classes). Missing a churner (false negative) costs more: the company loses the customer and all future payments, while a false alarm only costs one retention offer.
2. **Why do we split before scaling?** The scaler learns the mean and standard deviation. Fitted on all the data, it would carry information from the test set into training (data leakage) and make the test score too optimistic. I split first and keep the scaler inside a Pipeline, so it is fitted only on training data, even inside every cross-validation fold.
3. **What score does a model that knows nothing get?** The `DummyClassifier` (always "No churn") gets 73.5% accuracy, but precision, recall and F1 are 0: it catches none of the churners.
4. **Which model came out best and why?** After tuning, Random Forest (CV F1 0.635), Logistic Regression (0.633), XGBoost (0.632) and SVM (0.625) are practically tied: the gaps are smaller than the variation between folds. Churn here depends on a few strong, simple effects (contract type, tenure, fiber internet, payment method), which even a linear model captures. What helped most was class weighting (recall from 0.54 to 0.80); tuning added up to +0.02 F1 and feature engineering only ±0.003.
5. **Why this main metric?** F1 for the churn class. Accuracy is misleading on imbalanced data (73.5% for doing nothing), and recall alone could be maximised by flagging everyone. F1 is high only when I catch many churners *and* most alarms are real.
6. **Conclusion:** I recommend the tuned Random Forest with a decision threshold of 0.53. On the test set (used once) it reaches F1 **0.635** and recall **0.781**: it catches **292 of the 374** churners, against 0 for the dummy and 208 for a simple logistic regression (F1 0.601). **Limitation:** the data is a single snapshot without customer history (usage, support calls), and the threshold maximises F1 rather than the real business cost of each mistake.

## Recommended model

| Model (test set) | Churners caught (of 374) | Recall | Precision | F1 |
|---|---|---|---|---|
| Dummy (always "No") | 0 | 0.000 | 0.000 | 0.000 |
| Simple logistic regression | 208 | 0.556 | 0.654 | 0.601 |
| **Tuned Random Forest, threshold 0.53** | **292** | **0.781** | 0.536 | **0.635** |

Logistic regression (test F1 0.622) would be an equally defensible, simpler choice.
