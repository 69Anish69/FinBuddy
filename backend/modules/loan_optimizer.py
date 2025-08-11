import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- UI Function -------------------------
def display_loan_optimizer_ui():
    st.header("💳 Loan Optimizer")
    st.write("Plan the most efficient way to repay multiple loans using smart strategies.")

    with st.form("loan_input_form"):
        num_loans = st.number_input("Number of Loans", min_value=1, max_value=10, value=2, step=1)

        loans = []
        for i in range(num_loans):
            st.markdown(f"**Loan {i+1}**")
            principal = st.number_input(f"Loan {i+1} Principal ($)", min_value=0.0, step=100.0, key=f"p{i}")
            interest = st.number_input(f"Loan {i+1} Annual Interest Rate (%)", min_value=0.0, step=0.1, key=f"r{i}")
            min_payment = st.number_input(f"Loan {i+1} Minimum Monthly Payment ($)", min_value=0.0, step=10.0, key=f"m{i}")
            loans.append({"principal": principal, "interest": interest, "min_payment": min_payment})

        total_monthly_budget = st.number_input("Total Monthly Budget for Loan Repayment ($)", min_value=0.0, step=50.0)
        strategy = st.selectbox("Repayment Strategy", ["Avalanche (Highest Interest First)", "Snowball (Lowest Balance First)", "Equal Distribution"])

        submitted = st.form_submit_button("Optimize Repayment Plan")

    if submitted:
        result = simulate_loans(loans, total_monthly_budget, strategy)
        display_strategy_result(result, strategy)


# ----------------------- Loan Simulation -------------------------
def simulate_loans(loans, budget, strategy):
    df = pd.DataFrame(loans)
    df["balance"] = df["principal"]
    df["months"] = 0
    df["total_interest"] = 0.0

    plan = []

    month = 1
    while df["balance"].sum() > 0 and month < 300:
        payments = np.zeros(len(df))

        if strategy.startswith("Avalanche"):
            order = df.sort_values("interest", ascending=False).index
        elif strategy.startswith("Snowball"):
            order = df.sort_values("balance", ascending=True).index
        else:
            order = df.index

        remaining_budget = budget

        for i in order:
            if df.loc[i, "balance"] <= 0:
                continue

            min_pay = df.loc[i, "min_payment"]
            interest_month = df.loc[i, "balance"] * (df.loc[i, "interest"] / 12 / 100)
            pay_amount = min(min_pay, df.loc[i, "balance"] + interest_month)

            if remaining_budget >= pay_amount:
                payments[i] += pay_amount
                remaining_budget -= pay_amount
            else:
                payments[i] += remaining_budget
                remaining_budget = 0
                break

        # Extra distribution
        for i in order:
            if df.loc[i, "balance"] <= 0:
                continue

            extra_possible = df.loc[i, "balance"] + df.loc[i, "balance"] * (df.loc[i, "interest"] / 12 / 100) - payments[i]
            if remaining_budget > 0 and extra_possible > 0:
                add_pay = min(remaining_budget, extra_possible)
                payments[i] += add_pay
                remaining_budget -= add_pay

        interest_paid = []
        for i in df.index:
            if df.loc[i, "balance"] > 0:
                interest = df.loc[i, "balance"] * (df.loc[i, "interest"] / 12 / 100)
                df.loc[i, "total_interest"] += interest
                df.loc[i, "balance"] += interest - payments[i]
                df.loc[i, "balance"] = max(df.loc[i, "balance"], 0)
                df.loc[i, "months"] += 1
                interest_paid.append(interest)
            else:
                interest_paid.append(0)

        plan.append({"Month": month, **{f"Loan {i+1}": payments[i] for i in range(len(df))}})
        month += 1

    summary = df[["months", "total_interest"]].copy()
    summary["Loan"] = [f"Loan {i+1}" for i in df.index]

    return {"summary": summary, "plan": pd.DataFrame(plan)}


# ------------------------ Display ----------------------------
def display_strategy_result(result, strategy):
    st.subheader("📊 Strategy Summary")
    summary = result["summary"]
    st.dataframe(summary.set_index("Loan"))

    st.markdown("### 💡 Smart Recommendation")
    best = summary["total_interest"].idxmin()
    best_loan = summary.iloc[best]
    st.success(f"✅ The strategy '{strategy}' will help you finish **Loan {best+1}** in {int(best_loan['months'])} months, saving a total of ${best_loan['total_interest']:.2f} in interest.")

    st.markdown("---")
    st.subheader("📉 Monthly Repayment Plan")
    plan_df = result["plan"]
    st.line_chart(plan_df.set_index("Month"))

    with st.expander("📥 Download Repayment Plan"):
        csv = plan_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "loan_plan.csv", "text/csv")
