"""Simulate transaction data, then fit BG/NBD + Gamma-Gamma. Run: python clv_bgnbd.py"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data, summary_data_from_transaction_data
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

rng = np.random.default_rng(42)
START, END = pd.Timestamp("2023-10-27"), pd.Timestamp("2026-10-26")      # 3-year window
CAL_END = END - pd.Timedelta(days=365)                                   # last year held out
N = 5000

# ---------- 1. simulate from the BG/NBD + Gamma-Gamma generative story ----------
TRUE = dict(r=0.55, alpha=45.0, a=0.35, b=1.6, p=6.0, q=4.0, gamma=750.0)
lam = rng.gamma(TRUE["r"], 1 / TRUE["alpha"], N)               # purchase rate per day
drop = rng.beta(TRUE["a"], TRUE["b"], N)                       # dropout prob after each purchase
nu = rng.gamma(TRUE["q"], 1 / TRUE["gamma"], N)                # customer spend scale
acq = START + pd.to_timedelta(rng.integers(0, 700, N), "D")    # acquisition dates
rows = []
for i in range(N):
    t = (acq[i] - START).days + rng.random()
    horizon = (END - START).days
    times = [t]
    while True:
        t += rng.exponential(1 / lam[i])
        if t > horizon: break
        times.append(t)
        if rng.random() < drop[i]: break
    spend = np.maximum(rng.gamma(TRUE["p"], 1 / nu[i], len(times)), 20).round(2)
    for tm, s in zip(times, spend):
        rows.append((f"C{i+1:05d}", START + pd.to_timedelta(tm, "D"), s))
tx = pd.DataFrame(rows, columns=["customer_id", "purchase_ts", "order_value"])
tx["purchase_ts"] = tx.purchase_ts.dt.floor("s")
tx.sort_values(["customer_id", "purchase_ts"]).to_csv("transactions.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
tx["purchase_date"] = tx.purchase_ts.dt.normalize()
print("transactions:", len(tx), "customers:", tx.customer_id.nunique(), "mean order:", round(tx.order_value.mean(), 1))

# ---------- 2. calibration / holdout fit ----------
cal = calibration_and_holdout_data(tx, "customer_id", "purchase_date", calibration_period_end=CAL_END,
                                   observation_period_end=END, monetary_value_col="order_value")
bgf = BetaGeoFitter(penalizer_coef=0.001).fit(cal.frequency_cal, cal.recency_cal, cal.T_cal)
pred_hold = bgf.conditional_expected_number_of_purchases_up_to_time(cal.duration_holdout, cal.frequency_cal, cal.recency_cal, cal.T_cal)
act_hold = cal.frequency_holdout
bg_metrics = {"holdout_customers": len(cal), "actual_repeat_purchases": int(act_hold.sum()), "predicted_repeat_purchases": round(pred_hold.sum(), 0),
              "total_error_pct": round(100 * (pred_hold.sum() - act_hold.sum()) / act_hold.sum(), 1),
              "MAE_per_customer": round(mean_absolute_error(act_hold, pred_hold), 3), "corr": round(np.corrcoef(act_hold, pred_hold)[0, 1], 3)}
params = bgf.params_.round(3).to_dict()
# Gamma-Gamma: independence assumption + fit on repeat buyers
rep = cal[cal.frequency_cal > 0]
indep = round(np.corrcoef(rep.frequency_cal, rep.monetary_value_cal)[0, 1], 3)
ggf = GammaGammaFitter(penalizer_coef=0.001).fit(rep.frequency_cal, rep.monetary_value_cal)
hold_rep = cal[(cal.frequency_cal > 0) & (cal.frequency_holdout > 0)]
gg_pred = ggf.conditional_expected_average_profit(hold_rep.frequency_cal, hold_rep.monetary_value_cal)
gg = {"repeat_customers_cal": len(rep), "corr_freq_vs_monetary": indep, "gg_params": ggf.params_.round(2).to_dict(),
      "mean_actual_order_holdout": round(hold_rep.monetary_value_holdout.mean(), 1), "mean_predicted_order": round(gg_pred.mean(), 1),
      "mean_naive_order_cal": round(hold_rep.monetary_value_cal.mean(), 1)}

# 12-month CLV from calibration data -> compare with actual holdout spend
cal["clv_bgnbd"] = ggf.customer_lifetime_value(bgf, cal.frequency_cal, cal.recency_cal, cal.T_cal, cal.monetary_value_cal,
                                               time=12, discount_rate=0.0, freq="D") if False else np.nan
ex_purch = bgf.conditional_expected_number_of_purchases_up_to_time(365, cal.frequency_cal, cal.recency_cal, cal.T_cal)
exp_val = ggf.conditional_expected_average_profit(cal.frequency_cal, cal.monetary_value_cal)
exp_val = exp_val.where(cal.frequency_cal > 0, rep.monetary_value_cal.mean())   # no repeat history -> use population mean
cal["clv_bgnbd"] = ex_purch * exp_val
cal["actual_holdout_spend"] = cal.frequency_holdout * cal.monetary_value_holdout.replace(0, np.nan).fillna(0)
# ground truth for holdout spend: sum of order values in holdout window (repeat + any)
hold_tx = tx[(tx.purchase_date > CAL_END)].groupby("customer_id").order_value.sum()
cal["actual_holdout_spend"] = hold_tx.reindex(cal.index).fillna(0)
# the first purchase of a customer acquired in the holdout window is not a repeat, but they are not in cal; fine.

# ML comparison on the same task: predict holdout spend from calibration RFM features
feat = cal[["frequency_cal", "recency_cal", "T_cal", "monetary_value_cal"]]
idx = np.arange(len(cal)); rng.shuffle(idx); tr, te = idx[: int(.7 * len(idx))], idx[int(.7 * len(idx)):]
gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=.05, random_state=42).fit(feat.iloc[tr], cal.actual_holdout_spend.iloc[tr])
cal["clv_ml"] = gb.predict(feat)
t = cal.iloc[te]
def top_decile_capture(score):
    k = int(len(t) * .1); chosen = t.assign(s=score).nlargest(k, "s")
    return round(chosen.actual_holdout_spend.sum() / t.actual_holdout_spend.sum(), 3)
cmp_ = pd.DataFrame([
    {"model": "Naive (avg calibration spend per customer)", "R2": r2_score(t.actual_holdout_spend, np.full(len(t), cal.actual_holdout_spend.iloc[tr].mean())),
     "MAE": mean_absolute_error(t.actual_holdout_spend, np.full(len(t), cal.actual_holdout_spend.iloc[tr].mean())), "top10_capture": top_decile_capture(np.random.default_rng(1).random(len(t)))},
    {"model": "BG/NBD + Gamma-Gamma", "R2": r2_score(t.actual_holdout_spend, t.clv_bgnbd), "MAE": mean_absolute_error(t.actual_holdout_spend, t.clv_bgnbd), "top10_capture": top_decile_capture(t.clv_bgnbd)},
    {"model": "Gradient boosting on RFM", "R2": r2_score(t.actual_holdout_spend, t.clv_ml), "MAE": mean_absolute_error(t.actual_holdout_spend, t.clv_ml), "top10_capture": top_decile_capture(t.clv_ml)}]).round(3)
cmp_.to_csv("bgnbd_vs_ml.csv", index=False)

# ---------- 3. final model on all data -> 12-month CLV for every customer ----------
full = summary_data_from_transaction_data(tx, "customer_id", "purchase_date", monetary_value_col="order_value", observation_period_end=END)
bgf_f = BetaGeoFitter(penalizer_coef=0.001).fit(full.frequency, full.recency, full["T"])
repf = full[full.frequency > 0]
ggf_f = GammaGammaFitter(penalizer_coef=0.001).fit(repf.frequency, repf.monetary_value)
full["p_alive"] = bgf_f.conditional_probability_alive(full.frequency, full.recency, full["T"])
full["exp_purchases_12m"] = bgf_f.conditional_expected_number_of_purchases_up_to_time(365, full.frequency, full.recency, full["T"])
ev = ggf_f.conditional_expected_average_profit(full.frequency, full.monetary_value).where(full.frequency > 0, repf.monetary_value.mean())
full["exp_order_value"] = ev
full["clv_12m"] = full.exp_purchases_12m * ev / (1 + 0.10)       # 10% annual discount, applied once over the 12-month horizon
full["tier"] = pd.qcut(full.clv_12m.rank(method="first"), [0, .5, .8, .95, 1], labels=["Low (bottom 50%)", "Medium", "High", "Top 5%"])
def act(r):
    if r.p_alive < 0.3: return "Lapsed: low-cost reactivation"
    if r.tier == "Top 5%": return "VIP retention"
    if r.tier == "High": return "Loyalty / cross-sell"
    if r.frequency == 0: return "One-time buyer: second-purchase nudge"
    return "Standard nurture"
full["recommended_action"] = full.apply(act, axis=1)
full.round(3).to_csv("customer_clv_bgnbd.csv")
summ = full.groupby("recommended_action").agg(customers=("clv_12m", "size"), total_clv_12m=("clv_12m", "sum"), avg_clv=("clv_12m", "mean"), avg_p_alive=("p_alive", "mean")).round(2)
summ.to_csv("bgnbd_action_summary.csv")
share = {"top5pct": round(full.clv_12m.nlargest(250).sum() / full.clv_12m.sum(), 3), "top20pct": round(full.clv_12m.nlargest(1000).sum() / full.clv_12m.sum(), 3)}

# ---------- 4. figures ----------
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 4.2)); plot_calibration_purchases_vs_holdout_purchases(bgf, cal, n=7, ax=ax)
ax.set_title("BG/NBD: actual vs predicted purchases in holdout year"); fig.tight_layout(); fig.savefig("figures/07_bgnbd_calibration.png"); plt.close()
fig, ax = plt.subplots(figsize=(7, 4.2)); ax.hist(full.p_alive, bins=30, color="#3b6ea5"); ax.set(xlabel="P(customer still active)", ylabel="customers", title="Probability alive (BG/NBD)")
fig.tight_layout(); fig.savefig("figures/08_probability_alive.png"); plt.close()
fig, ax = plt.subplots(figsize=(7, 4.2)); ax.hist(full.clv_12m.clip(upper=full.clv_12m.quantile(.99)), bins=40, color="#3b6ea5")
ax.set(xlabel="12-month CLV (clipped at 99th percentile)", ylabel="customers", title="12-month CLV distribution (BG/NBD + Gamma-Gamma)")
fig.tight_layout(); fig.savefig("figures/09_bgnbd_clv.png"); plt.close()
fig, ax = plt.subplots(figsize=(7, 3.6)); (summ.total_clv_12m / 1e6).sort_values().plot.barh(ax=ax, color="#3b6ea5")
ax.set(xlabel="Total 12-month CLV (millions)", ylabel="", title="CLV by recommended action"); fig.tight_layout(); fig.savefig("figures/10_bgnbd_actions.png"); plt.close()

print("\nTRUE params:", TRUE); print("BG/NBD fit:", params); print("Gamma-Gamma:", gg); print("holdout:", bg_metrics)
print("\n", cmp_.to_string(index=False)); print("\n", summ); print("\nshare:", share)
print("per-customer tx stats:", tx.groupby("customer_id").size().describe().round(2).to_dict())
pd.Series({**bg_metrics, **{f"gg_{k}": v for k, v in gg.items()}, **share, "true_params": str(TRUE), "fit_params": str(params)}).to_csv("bgnbd_metrics.csv", header=["value"])
