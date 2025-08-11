import streamlit as st
import pandas as pd

from dateutil.relativedelta import relativedelta

# Utility function to detect anomalies
def detect_anomalies(df):
    df = df.copy()
    df['Month'] = pd.to_datetime(df['Month'])
    monthly = df.groupby(df['Month'].dt.to_period("M"))['Expense'].sum().reset_index()
    monthly['Month'] = monthly['Month'].dt.to_timestamp()
    mean = monthly['Expense'].mean()
    std = monthly['Expense'].std()
    threshold = mean + 1.5 * std
    anomalies = monthly[monthly['Expense'] > threshold]
    return anomalies

def display_anomaly_ui():
    st.subheader("🔍 Anomaly Detection in Expenses")
    st.markdown("<small>We use machine learning (statistical trends) to identify months where your expenses were unusually high.</small>", unsafe_allow_html=True)

    df = st.session_state.get("finance_df", None)
    if df is None:
        st.warning("📂 Please upload your financial data first.")
        return

    if "Month" not in df.columns or "Expense" not in df.columns:
        st.error("The dataset must contain 'Month' and 'Expense' columns.")
        return

    anomalies = detect_anomalies(df)
    if anomalies.empty:
        st.success("✅ No significant anomalies detected in your expense trends.")
        return

    st.markdown("### ⚠️ Detected Anomalies in Expenses")
    st.dataframe(anomalies)

    st.markdown("### 🧠 Strategy & Advice")

    used = set()
    for _, row in anomalies.iterrows():
        month_str = row['Month'].strftime("%B %Y")
        current_expense = int(row['Expense'])
        avg_expense = int(df['Expense'].mean())

        st.markdown(f"#### 📅 {month_str}")
        st.markdown(f"You spent **${current_expense}**, which is **above your monthly average (${avg_expense})**.")
        st.markdown("This month was detected as an anomaly based on your trend.")

        st.markdown("**🔍 Possible Causes:**")
        st.markdown("- One-time large purchase (e.g., travel, shopping, gadget)")
        st.markdown("- Lack of budgeting discipline")

        st.markdown("**✅ Recommended Actions:**")

        suggestions = []

        if "budgeting" not in used:
            suggestions.append("Create a strict monthly budget and split expenses into essentials vs discretionary.")
            used.add("budgeting")

        if "subscriptions" not in used:
            suggestions.append("Review recurring subscriptions and cancel unused ones.")
            used.add("subscriptions")

        if "delay-purchase" not in used:
            suggestions.append("Avoid impulse buying. Add to wishlist and wait 2–3 days before purchase.")
            used.add("delay-purchase")

        if "food-control" not in used:
            suggestions.append("Reduce food delivery. Cook more at home and limit weekend treats.")
            used.add("food-control")

        if "trip-bundling" not in used:
            suggestions.append("Bundle errands into fewer transport trips to save costs.")
            used.add("trip-bundling")

        if "emergency-fund" not in used:
            suggestions.append("Keep an emergency fund buffer (10–15% of income) for such unexpected months.")
            used.add("emergency-fund")

        for item in suggestions:
            st.markdown(f"- {item}")
