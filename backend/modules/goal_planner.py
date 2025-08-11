
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def display_goal_planner_ui():
    st.markdown("## 🧭 Set Your Financial Goal")

    df = st.session_state.get("finance_df", None)
    if df is None or df.empty:
        st.warning("Please upload financial data to use this feature.")
        return

    # Calculate savings
    df["Savings"] = df["Income"] - df["Expense"]
    df = df.sort_values("Month")
    avg_savings = df["Savings"].mean()
    st.success("✅ Monthly savings calculated successfully.")
    st.dataframe(df[["Month", "Income", "Expense", "Savings"]].head(12))

    # Select goal type
    goal_type = st.selectbox("🎯 Select Goal Type", ["Custom", "Emergency Fund", "Vacation", "Car Purchase", "Home Downpayment"])

    # Suggested default goal amounts based on type
    default_goals = {
        "Emergency Fund": 50000,
        "Vacation": 100000,
        "Car Purchase": 300000,
        "Home Downpayment": 800000,
        "Custom": 100000
    }

    goal_amount = st.number_input("💰 Goal Amount ($)", min_value=1000, step=1000, value=default_goals.get(goal_type, 100000))

    col1, col2 = st.columns(2)
    with col1:
        # Let user choose either months or target date
        use_date = st.checkbox("📅 Choose Target Date Instead")

    if use_date:
        with col2:
            target_date = st.date_input("Target Date", value=datetime.today())
            months = max(1, (target_date.year - df["Month"].dt.year.max()) * 12 + (target_date.month - df["Month"].dt.month.max()))
    else:
        with col2:
            months = st.slider("📆 Months to achieve goal", min_value=1, max_value=60, value=12)

    # Calculate projection
    projected_total = avg_savings * months
    required_monthly_savings = goal_amount / months

    st.markdown(f"### 📊 Projected Savings in {months} months: ${projected_total:,.2f}")
    st.info(f"💡 You need to save ${required_monthly_savings:,.2f} per month to reach ${goal_amount:,.0f} goal.")

    if projected_total >= goal_amount:
        st.success("🎯 Congratulations! Your goal is achievable with current savings.")
    else:
        st.warning("⚠️ You may fall short. Try increasing your savings rate or reducing expenses.")

    # Visual Chart
    month_range = list(range(1, months + 1))
    projected_savings = avg_savings * pd.Series(month_range)

    fig, ax = plt.subplots()
    ax.plot(month_range, projected_savings, label='Projected Savings', marker='o')
    ax.axhline(goal_amount, color='red', linestyle='--', label='Goal Amount')
    ax.set_xlabel("Months")
    ax.set_ylabel("$")
    ax.set_title("Goal Achievement Forecast")
    ax.legend()
    st.pyplot(fig)
