# Demo Script (about 2 minutes)

Record your screen with a narration (Win+G Game Bar, OBS, or PowerPoint screen recording). Keep a terminal and the `figures/` folder open.

| Time | Show | Say |
|---|---|---|
| 0:00-0:15 | Title slide or README | "I'm [name]. This is Project 54, Customer Lifetime Value prediction. The goal is to find which customers are worth the most over the next year and what to do for each group." |
| 0:15-0:35 | `dataset.csv`, then `data_quality_audit.csv` | "I started with the provided 500-customer file. My audit found it is synthetic: total spend equals frequency times recency in every row and the segment label is random. So its 0.947 R-squared is not trustworthy, and I say so in the report." |
| 0:35-0:55 | Run `python clv_pipeline.py`, open `figures/01` to `03` | "The script cleans the data, builds RFM segments and trains regression. These three charts show the skewed CLV, the correlations and the recency-frequency pattern." |
| 0:55-1:20 | Run `python clv_bgnbd.py` (or show its output), open `figures/07_bgnbd_calibration.png` | "The brief asks for BG/NBD, which needs purchase histories, so I generated 5,000 customers and about 31,000 transactions with 3% labelled anomalies. The anomaly rule found all of them. BG/NBD and Gamma-Gamma were trained on two years and tested on the third: predicted repeat purchases were within 3.2% of actual." |
| 1:20-1:40 | `bgnbd_vs_ml.csv`, `figures/10_bgnbd_actions.png` | "BG/NBD ranks customers slightly better than gradient boosting: the top 10 percent capture 52 percent of next-year spend against 47 percent. Customers are placed into five actions, such as VIP retention and low-cost reactivation. The top 5 percent hold about 42 percent of value." |
| 1:40-1:55 | MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db` | "Every run is tracked in MLflow with parameters and metrics." |
| 1:55-2:10 | Report, Limitations section | "Both datasets are synthetic, so these results validate the method and the code, not real-world accuracy. With real transaction data the same pipeline applies unchanged. No personal data is used, and fairness must be checked before real use. Thank you." |

Before recording: run both scripts once so the files are fresh, and close unrelated windows.
