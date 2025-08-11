
# modules/sms_parser.py

import streamlit as st
import re
import pandas as pd

def extract_transactions_from_text(text):
    # Simple regex patterns for amounts and descriptions
    pattern = r"(?:spent|debited|credited).*?\₹([0-9,]+).*?on (.*?)\b"
    matches = re.findall(pattern, text, re.IGNORECASE)

    transactions = []
    for amount, desc in matches:
        amt_clean = float(amount.replace(",", ""))
        transactions.append({"Description": desc.strip().title(), "Amount (₹)": amt_clean})

    return pd.DataFrame(transactions)

def display_sms_parser_ui():
    st.header("📩 Email/SMS Transaction Extractor")

    st.markdown("Paste bank/SMS notifications (from email, SMS, WhatsApp) to auto-extract transactions.")

    user_input = st.text_area("Paste Messages Here", height=200, placeholder="e.g. ₹500 spent on Amazon ₹1500 credited to your account on Flipkart")

    if st.button("Extract Transactions"):
        if not user_input.strip():
            st.warning("Please paste some content.")
            return

        df = extract_transactions_from_text(user_input)

        if df.empty:
            st.info("No transactions found. Try different message formats.")
        else:
            st.success(f"✅ Extracted {len(df)} transactions")
            st.dataframe(df)

            st.download_button("Download Extracted CSV", df.to_csv(index=False), "sms_transactions.csv", "text/csv")
