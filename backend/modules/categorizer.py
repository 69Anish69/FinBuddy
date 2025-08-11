import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Simple keyword-based tagging
CATEGORY_KEYWORDS = {
    "Food & Dining": ["zomato", "swiggy", "restaurant", "food", "cafe"],
    "Shopping": ["amazon", "flipkart", "myntra", "shopping", "store"],
    "Utilities": ["electricity", "water", "gas", "bill", "broadband"],
    "Travel": ["ola", "uber", "train", "flight", "travel"],
    "Health": ["pharmacy", "hospital", "clinic", "health"],
    "Entertainment": ["netflix", "hotstar", "spotify", "entertainment"],
    "Others": []
}

def tag_description(desc):
    desc = str(desc).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in desc for kw in keywords):
            return category
    return "Others"

def display_categorizer_ui():
    st.header("📂 Expense Categorization by Description")

    df = st.session_state.get("finance_df", None)
    if df is None:
        st.warning("Please upload your financial data first.")
        return

    if "Description" not in df.columns:
        st.error("The dataset must contain a 'Description' column to categorize expenses.")
        return

    df = df.copy()
    df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
    df.dropna(subset=["Month", "Expense"], inplace=True)
    df["Category"] = df["Description"].apply(tag_description)

    st.subheader("📊 Category-wise Total Expenses")
    cat_expense = df.groupby("Category")["Expense"].sum().sort_values(ascending=False)
    st.bar_chart(cat_expense)

    st.subheader("🥇 Top 3 Spending Categories + Tips")
    top3 = cat_expense.head(3)

    for category, amount in top3.items():
        percent = round((amount / df["Expense"].sum()) * 100, 1)
        st.markdown(f"""
        #### 🔹 {category}: ₹{int(amount)} ({percent}% of total)
        **Advice:**""")

        if category == "Shopping":
            st.markdown("""- Limit online shopping to 1-2 times a month.
- Use wishlist for delayed decisions.""")
        elif category == "Food & Dining":
            st.markdown("""- Cook more meals at home.
- Limit ordering food on weekends only.""")
        elif category == "Travel":
            st.markdown("""- Use public transport more often.
- Club trips to reduce frequent ride costs.""")
        elif category == "Health":
            st.markdown("""- Review if any non-essential health items were bought.
- Consider using company health insurance.""")
        elif category == "Utilities":
            st.markdown("""- Check for unnecessary usage (AC, geyser, etc.).
- Automate bills to avoid late fees.""")
        else:
            st.markdown("- Review these transactions manually to reassign or monitor.")

    # Pie Chart
    st.subheader("🧁 Expense Category Distribution")
    fig, ax = plt.subplots()
    cat_expense.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)
