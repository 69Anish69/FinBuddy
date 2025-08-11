
# modules/portfolio.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Simulate current market prices (in real-world use APIs like Yahoo Finance)
SIMULATED_PRICES = {
    "AAPL": 195,
    "GOOGL": 2800,
    "TCS": 3700,
    "INFY": 1450,
    "NIFTY50": 22000
}

def calculate_portfolio_metrics(df):
    df["Current Price"] = df["Symbol"].map(SIMULATED_PRICES)
    df["Current Value"] = df["Current Price"] * df["Quantity"]
    df["Investment"] = df["Buy Price"] * df["Quantity"]
    df["Profit/Loss"] = df["Current Value"] - df["Investment"]
    df["Return (%)"] = (df["Profit/Loss"] / df["Investment"]) * 100
    return df

def display_portfolio_ui():
    st.header("💼 Portfolio Tracker & Investment Suggestions")

    uploaded_file = st.file_uploader("Upload your investment portfolio CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if not all(col in df.columns for col in ["Symbol", "Quantity", "Buy Price"]):
                st.error("CSV must have columns: Symbol, Quantity, Buy Price")
                return

            df = calculate_portfolio_metrics(df)

            st.subheader("📊 Portfolio Summary")
            st.dataframe(df)

            total_investment = df["Investment"].sum()
            total_current = df["Current Value"].sum()
            total_profit = total_current - total_investment

            st.markdown(f"**Total Invested:** ₹{total_investment:,.2f}")
            st.markdown(f"**Current Value:** ₹{total_current:,.2f}")
            st.markdown(f"**Net Gain/Loss:** ₹{total_profit:,.2f}")

            # Pie chart for allocation
            st.subheader("📈 Asset Allocation")
            allocation = df.groupby("Symbol")["Current Value"].sum()
            fig, ax = plt.subplots()
            ax.pie(allocation, labels=allocation.index, autopct="%1.1f%%")
            st.pyplot(fig)

            # Download report
            st.download_button("Download Portfolio Report", df.to_csv(index=False), "portfolio_summary.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing portfolio file: {e}")
