# Customer Lifetime Value (CLV) Prediction

HCLTech Career Shaper Programme, III Year Industry Project 2025-26 (Project 54, Data Science / Python).

Two parts:

1. **`clv_pipeline.py`** - audit, cleaning, EDA, RFM segmentation and regression on the supplied 500-customer `dataset.csv`.
   The audit found that this file is synthetic (`total_spend == frequency * recency_days` in every row, random `segment` label),
   so its high regression score (R2 0.947 on log CLV) reflects that formula, not real behaviour.
2. **`clv_bgnbd.py`** - simulates 5,000 customers / 30,318 transactions over 3 years, fits BG/NBD and Gamma-Gamma (Lifetimes),
   validates on a held-out final year and compares with gradient boosting.

Because the transaction data are simulated from the model family being fitted, results validate the method and code, not performance on real customers.

## Key results (held-out year)
| Check | Result |
|---|---|
| Repeat purchases, predicted vs actual | 10,789 vs 10,445 (+3.3%) |
| Avg order value, predicted vs actual | 1,558 vs 1,512 (+3.1%) |
| R2 on next-year spend | BG/NBD + Gamma-Gamma 0.574, gradient boosting 0.523 |
| Top-10% capture of next-year spend | 50.4% vs 47.3% |

## Run
```
pip install -r requirements.txt
python clv_pipeline.py
python clv_bgnbd.py
```

Deliverables: `CLV_Project_Report.docx`, `Project_Synopsis_CLV_Prediction.docx`, `data_dictionary.csv`, `figures/`.
