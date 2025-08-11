
# modules/tax.py

import streamlit as st
import pandas as pd

def calculate_tax(income):
    if income <= 250000:
        return 0
    elif income <= 500000:
        return (income - 250000) * 0.05
    elif income <= 1000000:
        return (250000 * 0.05) + (income - 500000) * 0.2
    else:
        return (250000 * 0.05) + (500000 * 0.2) + (income - 1000000) * 0.3

def display_tax_ui():
    st.header("🧾 Tax Estimator (India - Basic Slabs)")

    st.markdown("Estimate your annual tax based on income")

    gross_income = st.number_input("Annual Gross Income (₹)", min_value=0.0, step=1000.0)
    deductions = st.number_input("Total Deductions (₹)", min_value=0.0, step=1000.0)

    if st.button("Estimate Tax"):
        taxable_income = max(0, gross_income - deductions)
        tax = calculate_tax(taxable_income)

        st.markdown(f"**Taxable Income:** ₹{taxable_income:,.2f}")
        st.markdown(f"**Estimated Tax Payable:** ₹{tax:,.2f}")
