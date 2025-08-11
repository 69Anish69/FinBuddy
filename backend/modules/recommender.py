
import streamlit as st
import pandas as pd

def display_recommendation_ui():
    st.header("📊 Personalized Financial Recommendations")

    df = st.session_state.get("finance_df", None)
    if df is None:
        st.warning("Please upload your financial data first.")
        return

    required_cols = {"Month", "Income", "Expense"}
    if not required_cols.issubset(df.columns):
        st.error("Missing required columns: Month, Income, Expense.")
        return

    df = df.copy()
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df.dropna(subset=["Month"], inplace=True)
    df.sort_values("Month", inplace=True)

    st.markdown("Based on your past financial behavior, here are personalized improvement suggestions:")

    try:
        # Recommendation 1: Savings Ratio
        df["Savings"] = df["Income"] - df["Expense"]
        df["SavingsRate"] = df["Savings"] / df["Income"]
        avg_savings_rate = df["SavingsRate"].mean()

        if avg_savings_rate < 0.1:
            st.error("🔻 Your average savings rate is very low (less than 10%).")
            st.markdown("""
            **Strategies:**
            - Aim to save at least 20% of your income monthly.
            - Categorize fixed vs flexible expenses and cut from flexible ones.
            - Automate savings into a separate account on payday.
            """)
        elif avg_savings_rate < 0.2:
            st.warning("⚠️ You're saving moderately (~10–20%).")
            st.markdown("""
            **Tips:**
            - Optimize your subscriptions and utility usage.
            - Consolidate small investments into a larger SIP.
            """)
        else:
            st.success("✅ Great! You're saving over 20% of your income.")
            st.markdown("""
            **Suggestions:**
            - Consider stepping up your SIP/investments annually.
            - Start planning for long-term goals (retirement, home).
            """)

        # Recommendation 2: Expense Volatility
        std_expense = df["Expense"].std()
        mean_expense = df["Expense"].mean()
        volatility_ratio = std_expense / mean_expense

        if volatility_ratio > 0.4:
            st.error("📉 Your monthly expenses are highly inconsistent.")
            st.markdown("""
            **Strategies:**
            - Identify months with spikes (see anomaly tab).
            - Set monthly spending targets and track weekly.
            - Use alerts to stay within budget.
            """)
        else:
            st.success("📊 Your expenses are fairly consistent.")

        # Recommendation 3: High Spending Flag (if monthly expense > 80% of income)
        high_spend_months = df[df["Expense"] > 0.8 * df["Income"]]
        if not high_spend_months.empty:
            st.error("💸 In some months, expenses exceeded 80% of income.")
            st.markdown(f"""
            **Concern:** You spent too much in these months:
            {', '.join(high_spend_months['Month'].dt.strftime('%b %Y').tolist())}

            **Suggestions:**
            - Avoid EMIs or credit purchases unless necessary.
            - Track your spending categories to identify overspending.
            - Consider envelope budgeting or 50/30/20 method.
            """)
        else:
            st.success("💰 Your expenses stayed within healthy limits.")

    except Exception as e:
        st.error(f"❌ Recommendation generation failed: {e}")
