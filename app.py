"""Streamlit dashboard for the CLV prediction project. Run: streamlit run app.py"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import streamlit as st
from lifetimes import BetaGeoFitter, GammaGammaFitter

BASE = Path(__file__).parent
st.set_page_config(page_title="CLV Prediction", page_icon="📈", layout="wide")


@st.cache_data
def load():
    cust = pd.read_csv(BASE / "customer_clv_bgnbd.csv")
    return (cust, pd.read_csv(BASE / "bgnbd_vs_ml.csv"), pd.read_csv(BASE / "anomaly_detection_metrics.csv", index_col=0).value,
            pd.read_csv(BASE / "data_quality_audit.csv", index_col=0).value, pd.read_csv(BASE / "model_metrics.csv"),
            pd.read_csv(BASE / "bgnbd_action_summary.csv"))


@st.cache_resource
def fit(cust):
    bgf = BetaGeoFitter(penalizer_coef=0.001).fit(cust.frequency, cust.recency, cust["T"])
    rep = cust[cust.frequency > 0]
    ggf = GammaGammaFitter(penalizer_coef=0.001).fit(rep.frequency, rep.monetary_value)
    return bgf, ggf, rep.monetary_value.mean()


cust, cmp_, anom, audit, ml500, actions = load()
bgf, ggf, pop_mean = fit(cust)

st.title("Customer Lifetime Value (CLV) Prediction")
st.caption("HCLTech Career Shaper Programme · Project 54 · Atharv Dhiman (2300320130071). "
           "Both datasets are synthetic, so results validate the method, not real-world accuracy.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Customer explorer", "Predict a customer", "Model validation", "Data quality"])

with tab1:
    a, b, c, d = st.columns(4)
    a.metric("Customers", f"{len(cust):,}")
    b.metric("Total 12-month CLV", f"{cust.clv_12m.sum() / 1e6:.1f} M")
    top = cust.clv_12m.nlargest(250).sum() / cust.clv_12m.sum()
    c.metric("Top 250 customers' share of CLV (top 5%)", f"{top:.1%}")
    d.metric("Probably lapsed (P(alive) < 0.3)", f"{(cust.p_alive < .3).mean():.0%}")
    left, right = st.columns(2)
    s = actions.set_index("recommended_action")
    left.subheader("Total 12-month CLV by retention action")
    left.bar_chart(s.total_clv_12m)
    right.subheader("Customers per action")
    right.bar_chart(s.customers)
    st.dataframe(s.rename(columns={"total_clv_12m": "Total CLV", "avg_clv": "Avg CLV", "avg_p_alive": "Avg P(alive)"}), use_container_width=True)
    st.image(str(BASE / "figures" / "09_bgnbd_clv.png"), caption="12-month CLV distribution")

with tab2:
    f1, f2, f3 = st.columns(3)
    act = f1.multiselect("Recommended action", sorted(cust.recommended_action.unique()))
    tier = f2.multiselect("CLV tier", cust.tier.unique().tolist())
    q = f3.text_input("Customer ID contains", "")
    v = cust.copy()
    if act: v = v[v.recommended_action.isin(act)]
    if tier: v = v[v.tier.isin(tier)]
    if q: v = v[v.customer_id.str.contains(q, case=False)]
    st.write(f"{len(v):,} customers")
    st.dataframe(v.sort_values("clv_12m", ascending=False), use_container_width=True, height=420)
    st.download_button("Download filtered customers (CSV)", v.to_csv(index=False), "customers_filtered.csv", "text/csv")

with tab3:
    st.write("Enter a customer's purchase history. The fitted BG/NBD and Gamma-Gamma models estimate whether they are still active and their 12-month value.")
    p1, p2, p3, p4 = st.columns(4)
    T = p3.number_input("Customer age T (days)", 1, 1100, 700)
    freq = p1.number_input("Repeat purchases", 0, 150, 3)
    rec = p2.number_input("Days between first and last purchase", 0, int(T), min(300, int(T)))
    mon = p4.number_input("Avg order value (repeat purchases)", 0.0, 50000.0, 1500.0, disabled=freq == 0)
    horizon = st.slider("Forecast horizon (days)", 30, 730, 365)
    palive = float(bgf.conditional_probability_alive(freq, rec, T))
    exp_p = float(bgf.conditional_expected_number_of_purchases_up_to_time(horizon, freq, rec, T))
    ev = float(ggf.conditional_expected_average_profit(freq, mon)) if freq > 0 else float(pop_mean)
    x, y, z = st.columns(3)
    x.metric("Probability still active", f"{palive:.1%}")
    y.metric(f"Expected purchases (next {horizon} days)", f"{exp_p:.2f}")
    z.metric("Expected value of this customer", f"{exp_p * ev / 1.10:,.0f}")
    if freq == 0:
        st.info("No repeat purchase yet: BG/NBD cannot tell whether this customer has left, so the average order value of repeat buyers is used.")

with tab4:
    st.subheader("BG/NBD + Gamma-Gamma vs gradient boosting (holdout year)")
    st.dataframe(cmp_.rename(columns={"top10_capture": "Top-10% capture"}), use_container_width=True, hide_index=True)
    st.image(str(BASE / "figures" / "07_bgnbd_calibration.png"), caption="Actual vs predicted purchases in the held-out year")
    st.subheader("Regression on the original 500-customer file")
    st.dataframe(ml500, use_container_width=True, hide_index=True)
    st.warning("The 0.947 R² on the 500-row file comes from its synthetic structure (total_spend = frequency × recency_days), not real predictive power.")

with tab5:
    st.subheader("Audit of the supplied 500-row dataset")
    st.dataframe(audit.rename("value"), use_container_width=True)
    st.subheader("Anomaly detection on the transaction data")
    st.dataframe(anom.rename("value"), use_container_width=True)
    st.image(str(BASE / "figures" / "06_action_value.png"), caption="500-customer file: CLV by recommended action")
