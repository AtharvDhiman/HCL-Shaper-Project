# Customer Lifetime Value (CLV) Prediction

HCLTech Career Shaper Programme, III Year Industry Project 2025-26 (Project 54, Data Science / Python).

Two parts:

1. **`clv_pipeline.py`** - audit, cleaning, EDA, RFM segmentation and regression on the supplied 500-customer `dataset.csv`.
   The audit found that this file is synthetic (`total_spend == frequency * recency_days` in every row, random `segment` label),
   so its high regression score (R2 0.947 on log CLV) reflects that formula, not real behaviour.
2. **`clv_bgnbd.py`** - simulates 5,000 customers / 31,256 transactions over 3 years (3% labelled anomalies, detected and removed),
   fits BG/NBD and Gamma-Gamma (Lifetimes), validates on a held-out final year and compares with gradient boosting.
   Runs are tracked with MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

Because the transaction data are simulated from the model family being fitted, results validate the method and code, not performance on real customers.

## Key results (held-out year)
| Check | Result |
|---|---|
| Repeat purchases, predicted vs actual | 10,765 vs 10,432 (+3.2%) |
| Avg order value, predicted vs actual | 1,534 vs 1,498 (+2.5%) |
| R2 on next-year spend | BG/NBD + Gamma-Gamma 0.606, gradient boosting 0.513 |
| Top-10% capture of next-year spend | 52.2% vs 47.1% |
| Anomaly detection (3% injected) | recall 1.00, precision 0.95 |

## Run
```
pip install -r requirements.txt
python clv_pipeline.py
python clv_bgnbd.py
```

Deliverables: `CLV_Project_Report.docx`, `Project_Synopsis_CLV_Prediction.docx`, `data_dictionary.csv` and `data_dictionary_transactions.csv`, `DEMO_SCRIPT.md`, `figures/`.
