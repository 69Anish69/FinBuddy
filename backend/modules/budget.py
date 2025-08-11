import streamlit as st
import pandas as pd

# Define example user budgets per category
DEFAULT_BUDGETS = {
    "Food & Dining": 4000,
    "Shopping": 3000,
    "Utilities": 2500,
    "Travel": 2000,
    "Health": 1500,
    "Entertainment": 1000,
    "Others": 1500
}

def display_budget_ui():
    st.header("💳 Budget vs Actual Tracker")

    df = st.session_state.get("finance_df", None)
    if df is None:
        st.warning("Please upload your financial data first.")
        return

    if "Category" not in df.columns or "Expense" not in df.columns:
        st.error("Missing required 'Category' and 'Expense' columns.")
        return

    st.markdown("### 🗓️ Monthly Summary")
    month = st.selectbox("Select Month", df["Month"].dt.strftime('%B %Y').unique()[::-1])
    df_month = df[df["Month"].dt.strftime('%B %Y') == month]

    if df_month.empty:
        st.info("No data available for selected month.")
        return

    actuals = df_month.groupby("Category")["Expense"].sum().to_dict()
    budget_vs_actual = []

    for cat, budget in DEFAULT_BUDGETS.items():
        spent = actuals.get(cat, 0)
        status = "✅ Within Limit" if spent <= budget else "🔺 Over Budget"
        budget_vs_actual.append({
            "Category": cat,
            "Budget (₹)": budget,
            "Spent (₹)": int(spent),
            "Status": status
        })

    st.dataframe(pd.DataFrame(budget_vs_actual), use_container_width=True)

    st.markdown("### 💡 Advice")
    for row in budget_vs_actual:
        if row["Status"] == "🔺 Over Budget":
            over_amount = row["Spent (₹)"] - row["Budget (₹)"]
            st.error(f"**{row['Category']}** exceeded by ₹{over_amount}. Try reducing spend in this area next month.")
        elif row["Spent (₹)"] == 0:
            st.warning(f"**{row['Category']}** has no spending this month. Verify data accuracy.")
        else:
            st.success(f"**{row['Category']}** is within budget. Good job!")
