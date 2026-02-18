import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from src.features import create_features

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="FactoryGuard AI",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 FactoryGuard AI - Enterprise Failure Intelligence")

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
model = joblib.load("models/factoryguard_model.pkl")

uploaded = st.file_uploader("📂 Upload Machine Sensor Data (CSV)", type=["csv"])

if uploaded:
    with st.spinner("Processing data..."):
        df = pd.read_csv(uploaded, parse_dates=["timestamp"])
        df = create_features(df)
        df = df.bfill().ffill().dropna()

        latest = df.sort_values("timestamp").groupby("machine_id").tail(1)

        X = latest.drop(columns=["machine_id", "timestamp", "failure"])
        preds = model.predict_proba(X)[:, 1]
        latest["Failure Risk"] = preds

    # ---------------------------------------------------
    # CREATE TABS
    # ---------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏠 Overview", "📊 Analytics", "🧠 Explainability", "🔄 Simulation"]
    )

    # ===================================================
    # OVERVIEW TAB
    # ===================================================
    with tab1:

        high = (latest["Failure Risk"] > 0.7).sum()
        medium = ((latest["Failure Risk"] > 0.4) & (latest["Failure Risk"] <= 0.7)).sum()
        low = (latest["Failure Risk"] <= 0.4).sum()
        avg = latest["Failure Risk"].mean()

        # 🚨 ALERT
        if high > 0:
            st.error(f"🚨 ALERT: {high} machines at HIGH RISK (>70%)")
        else:
            st.success("✅ All machines operating within safe limits")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔴 High Risk", high)
        col2.metric("🟠 Medium Risk", medium)
        col3.metric("🟢 Low Risk", low)
        col4.metric("📊 Avg Risk", f"{avg:.2%}")

        st.divider()

        # Bar Chart
        fig_bar = px.bar(
            latest,
            x="machine_id",
            y="Failure Risk",
            color="Failure Risk",
            color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
            template="plotly_dark",
            title="Machine Risk Levels"
        )
        st.plotly_chart(fig_bar, width="stretch")

        # Pie Chart
        risk_df = pd.DataFrame({
            "Risk Level": ["High", "Medium", "Low"],
            "Count": [high, medium, low]
        })

        fig_pie = px.pie(
            risk_df,
            names="Risk Level",
            values="Count",
            template="plotly_dark",
            color="Risk Level",
            color_discrete_map={
                "High": "#e74c3c",
                "Medium": "#f39c12",
                "Low": "#2ecc71"
            }
        )
        st.plotly_chart(fig_pie, width="stretch")

        # Download Button
        st.download_button(
            "📤 Download Predictions",
            latest.to_csv(index=False),
            file_name="machine_risk_predictions.csv",
            mime="text/csv"
        )

    # ===================================================
    # ANALYTICS TAB
    # ===================================================
    with tab2:

        st.subheader("Risk Distribution Analysis")

        fig_hist = px.histogram(
            latest,
            x="Failure Risk",
            nbins=20,
            template="plotly_dark",
            color_discrete_sequence=["#00BFFF"],
            title="Failure Risk Probability Distribution"
        )
        st.plotly_chart(fig_hist, width="stretch")

        sorted_df = latest.sort_values("Failure Risk", ascending=False)

        fig_line = px.line(
            sorted_df,
            x="machine_id",
            y="Failure Risk",
            markers=True,
            template="plotly_dark",
            title="Risk Ranking Across Machines"
        )
        st.plotly_chart(fig_line, width="stretch")

    # ===================================================
    # EXPLAINABILITY TAB
    # ===================================================
    with tab3:

        st.subheader("SHAP Model Explainability")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(fig)

    # ===================================================
    # SIMULATION TAB
    # ===================================================
    with tab4:

        st.subheader("Real-Time Risk Simulation")

        if st.button("🔄 Simulate New Readings"):
            simulated = latest.copy()
            simulated["Failure Risk"] = simulated["Failure Risk"] * (
                0.8 + 0.4 * np.random.rand(len(simulated))
            )

            fig_sim = px.bar(
                simulated,
                x="machine_id",
                y="Failure Risk",
                color="Failure Risk",
                template="plotly_dark",
                color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
                title="Simulated Risk Update"
            )

            st.plotly_chart(fig_sim, width="stretch")
