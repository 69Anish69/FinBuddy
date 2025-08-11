import pandas as pd
import streamlit as st
from prophet import Prophet
import matplotlib.pyplot as plt

def display_forecast_ui():
    st.header("📈 Income & Expense Forecast + Strategy")

    df = st.session_state.get("finance_df", None)
    if df is None:
        st.warning("Please upload your financial data in the sidebar.")
        return

    required_cols = {"Month", "Income", "Expense"}
    if not required_cols.issubset(df.columns):
        st.error("Missing required columns: Month, Income, Expense.")
        return

    df = df.copy()
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df.dropna(subset=["Month"], inplace=True)
    df.sort_values("Month", inplace=True)

    forecast_target = st.selectbox("📊 What would you like to forecast?", ["Income", "Expense"])
    horizon = st.slider("🔮 Forecast Months Into Future", 1, 12, 6)

    try:
        prophet_df = df[["Month", forecast_target]].rename(columns={"Month": "ds", forecast_target: "y"})

        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=horizon, freq="M")
        forecast = model.predict(future)

        st.subheader(f"📉 Forecast for {forecast_target} (next {horizon} months)")
        fig = model.plot(forecast)
        st.pyplot(fig)

        # Strategy section
        recent_value = df[forecast_target].iloc[-1]
        future_values = forecast.tail(horizon)[["ds", "yhat"]]

        st.subheader("🧠 AI Strategy & Financial Recommendation")

        avg_proj = future_values["yhat"].mean()
        growth = avg_proj - recent_value

        if forecast_target == "Income":
            if growth < 0:
                st.error("⚠️ Your projected income is decreasing.")
                st.markdown(f"""
                **Problem:** Based on your trend, your income may drop by ₹{abs(int(growth))} per month.

                **Strategies:**
                - Start building a 2-month emergency fund now.
                - Explore side income sources like freelancing or selling unused items.
                - Avoid any new major expenses until income stabilizes.
                """)
            else:
                st.success("✅ Your projected income is increasing.")
                st.markdown(f"""
                **Opportunity:** Your income may increase by ₹{int(growth)} on average.

                **Strategies:**
                - Allocate this surplus into high-interest savings or SIPs.
                - Boost your emergency fund or start a new investment goal.
                - Keep fixed expenses stable to take full advantage.
                """)
        else:  # Expense forecast
            if growth > 0:
                st.error("⚠️ Your projected expenses are rising.")
                st.markdown(f"""
                **Problem:** You may overspend by {int(growth)} per month if the trend continues.

                **Strategies:**
                - Analyze spending on food, shopping, and subscriptions. Cut 15–20%.
                - Set category-wise budgets and track weekly.
                - Avoid impulsive purchases this month.
                """)
            else:
                st.success("✅ Your expenses are projected to reduce.")
                st.markdown(f"""
                **Opportunity:** You are reducing expenses by ₹{abs(int(growth))} per month.

                **Strategies:**
                - Stay disciplined. Don't let occasional spikes break the streak.
                - Redirect the saved amount to a short-term goal or emergency fund.
                - Review recurring payments and cancel unnecessary ones.
                """)

    except Exception as e:
        st.error(f"❌ Forecasting error: {e}")
