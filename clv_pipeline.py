"""CLV prediction: cleaning, EDA, RFM segmentation, regression. Run: python clv_pipeline.py"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

AS_OF = pd.Timestamp("2026-10-26")
d = pd.read_csv("dataset.csv", parse_dates=["last_purchase_date"])

# ---- data quality audit ----
audit = {
    "rows": len(d),
    "missing_cells": int(d.isna().sum().sum()),
    "duplicate_ids": int(d.customer_id.duplicated().sum()),
    "total_spend == frequency*recency_days (%)": round(100 * (d.total_spend == d.frequency * d.recency_days).mean(), 1),
    "last_purchase_date after as-of date": int((d.last_purchase_date > AS_OF).sum()),
    "recency_days inconsistent with date (>30d)": int(((AS_OF - d.last_purchase_date).dt.days - d.recency_days).abs().gt(30).sum()),
}
seg_gap = d.groupby("segment")[["frequency", "recency_days", "clv_predicted"]].mean().round(1)
pd.Series(audit).to_csv("data_quality_audit.csv", header=["value"])

# ---- cleaning ----
d["date_valid"] = d.last_purchase_date <= AS_OF       # flag, don't drop: recency_days is the model input
d["recency_days"] = d.recency_days.clip(lower=0)
d["log_clv"] = np.log1p(d.clv_predicted)

# ---- RFM segmentation (replaces the label in the raw file, which is unrelated to behaviour) ----
r = pd.qcut(-d.recency_days, 4, labels=[1, 2, 3, 4]).astype(int)
f = pd.qcut(d.frequency.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
m = pd.qcut(d.total_spend.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
d["rfm_score"] = r + f + m
d["value_tier"] = pd.cut(d.clv_predicted.rank(pct=True), [0, .25, .5, .75, .9, 1],
                         labels=["Low", "Medium", "High", "Very High", "Top 10%"])
def strategy(row):
    if row.recency_days > 700: return "Win-back campaign"
    if row.value_tier in ("Top 10%", "Very High"): return "VIP loyalty / retention"
    if row.value_tier == "Low": return "Low-cost automated nurture"
    return "Cross-sell / upsell"
d["recommended_action"] = d.apply(strategy, axis=1)

# ---- modelling: predict log CLV from behavioural features ----
# total_spend is excluded from X: it is exactly frequency*recency_days here, and
# correlates 0.92 with clv_predicted, so it would leak the target.
X = d[["frequency", "recency_days", "avg_order_value"]].copy()
X["freq_x_rec"] = X.frequency * X.recency_days
y = d.log_clv
cv = KFold(5, shuffle=True, random_state=42)
models = {
    "Ridge": make_pipeline(StandardScaler(), Ridge(1.0)),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=.05, random_state=42),
}
rows = []
for name, mdl in models.items():
    p = cross_val_predict(mdl, X, y, cv=cv)
    pc = np.expm1(p)
    rows.append({"model": name, "R2_log": r2_score(y, p), "R2_currency": r2_score(d.clv_predicted, pc),
                 "MAE": mean_absolute_error(d.clv_predicted, pc),
                 "RMSE": mean_squared_error(d.clv_predicted, pc) ** .5})
base = np.full(len(d), d.clv_predicted.mean())
rows.append({"model": "Mean baseline", "R2_log": 0, "R2_currency": 0,
             "MAE": mean_absolute_error(d.clv_predicted, base), "RMSE": mean_squared_error(d.clv_predicted, base) ** .5})
res = pd.DataFrame(rows).round(3); res.to_csv("model_metrics.csv", index=False)
best = "GradientBoosting"
d["clv_model_pred"] = np.expm1(cross_val_predict(models[best], X, y, cv=cv)).round(2)
models[best].fit(X, y)
imp = pd.Series(models[best].feature_importances_, X.columns).sort_values()

# ---- figures ----
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(d.clv_predicted, bins=40, color="#3b6ea5"); ax[0].set(title="CLV distribution (right-skewed)", xlabel="clv_predicted")
ax[1].hist(d.log_clv, bins=40, color="#3b6ea5"); ax[1].set(title="log(1+CLV)", xlabel="log CLV")
fig.tight_layout(); fig.savefig("figures/01_clv_distribution.png"); plt.close()

c = d[["frequency", "recency_days", "avg_order_value", "total_spend", "clv_predicted"]].corr()
fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(5), c.columns, rotation=45, ha="right"); ax.set_yticks(range(5), c.columns)
for i in range(5):
    for j in range(5): ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Correlation matrix"); fig.colorbar(im); fig.tight_layout(); fig.savefig("figures/02_correlation.png"); plt.close()

fig, ax = plt.subplots(figsize=(6, 4.5))
sc = ax.scatter(d.recency_days, d.frequency, c=d.log_clv, cmap="viridis", s=14)
ax.set(xlabel="recency_days", ylabel="frequency", title="Recency vs frequency, coloured by log CLV"); fig.colorbar(sc)
fig.tight_layout(); fig.savefig("figures/03_recency_frequency.png"); plt.close()

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(d.clv_predicted, d.clv_model_pred, s=10, alpha=.6); lim = d.clv_predicted.max()
ax.plot([0, lim], [0, lim], "r--"); ax.set(xlabel="actual clv_predicted", ylabel="cross-validated prediction", title="Model fit (5-fold CV)")
fig.tight_layout(); fig.savefig("figures/04_actual_vs_predicted.png"); plt.close()

fig, ax = plt.subplots(figsize=(6, 3.5)); imp.plot.barh(ax=ax, color="#3b6ea5"); ax.set_title("Feature importance")
fig.tight_layout(); fig.savefig("figures/05_feature_importance.png"); plt.close()

act = d.groupby("recommended_action").agg(customers=("customer_id", "count"), total_clv=("clv_predicted", "sum"), avg_clv=("clv_predicted", "mean")).round(0)
fig, ax = plt.subplots(figsize=(7, 3.5)); (act.total_clv / 1e6).plot.barh(ax=ax, color="#3b6ea5")
ax.set(xlabel="Total CLV (millions)", ylabel="", title="CLV by recommended action"); fig.tight_layout(); fig.savefig("figures/06_action_value.png"); plt.close()

d.drop(columns=["log_clv"]).to_csv("dataset_clean.csv", index=False)
act.to_csv("segment_summary.csv")
print(pd.Series(audit).to_string()); print("\nRaw segment label vs behaviour:\n", seg_gap)
print("\n", res.to_string(index=False)); print("\n", imp.round(3).to_string()); print("\n", act)
