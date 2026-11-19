import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AfriScore | Credit Risk Assessment", page_icon="🏦", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("credit_risk_model.pkl")

bundle = load_model()
pipeline = bundle["pipeline"]

st.title("🏦 AfriScore: Micro-Loan Credit Risk Engine")
st.markdown("Evaluate borrower default risk using historical repayment velocity, loan terms, and applicant profile.")
st.divider()

with st.form("loan_form"):
    st.subheader("1. Current Loan Request")
    col1, col2 = st.columns(2)
    with col1:
        loanamount = st.number_input("Requested Loan Amount (₦)", min_value=5000, max_value=100000, value=20000, step=5000)
        termdays = st.selectbox("Loan Term (Days)", options=[15, 30, 60, 90], index=1)
    with col2:
        totaldue = st.number_input("Total Repayment Due (₦)", min_value=5000, max_value=120000, value=int(loanamount * 1.15), step=500)
        loannumber = st.number_input("Current Loan Sequence Number", min_value=1, max_value=30, value=3)

    st.subheader("2. Applicant Demographics")
    col3, col4 = st.columns(2)
    with col3:
        age = st.number_input("Applicant Age (Years)", min_value=18, max_value=75, value=32)
        bank_account_type = st.selectbox("Bank Account Type", options=bundle["account_types"])
    with col4:
        employment_status = st.selectbox("Employment Status", options=bundle["employment_statuses"])

    st.subheader("3. Historical Repayment Behavior")
    col5, col6 = st.columns(2)
    with col5:
        previous_loan_count = st.number_input("Completed Previous Loans", min_value=0, max_value=30, value=2)
        previous_average_loan = st.number_input("Average Previous Loan Amount (₦)", min_value=0, max_value=100000, value=15000, step=5000)
    with col6:
        late_payment_count = st.number_input("Number of Past Late Repayments", min_value=0, max_value=30, value=0)
        avg_days_late = st.number_input("Avg. Days Late (- = Early, + = Late)", min_value=-30.0, max_value=60.0, value=-2.0, step=0.5)

    submitted = st.form_submit_button("Assess Credit Risk", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([{
        "loanamount": loanamount,
        "totaldue": totaldue,
        "termdays": termdays,
        "loannumber": loannumber,
        "age": age,
        "bank_account_type": bank_account_type,
        "employment_status_clients": employment_status,
        "previous_loan_count": previous_loan_count,
        "previous_average_loan": previous_average_loan,
        "late_payment_count": late_payment_count,
        "avg_days_late": avg_days_late
    }])

    default_prob = pipeline.predict_proba(input_df)[0][1]

    st.divider()
    st.subheader("Credit Risk Assessment Result")
    m1, m2 = st.columns(2)
    m1.metric(label="Estimated Default Probability", value=f"{default_prob * 100:.1f}%")
    m2.metric(label="Repayment Confidence", value=f"{(1 - default_prob) * 100:.1f}%")
    st.progress(float(default_prob))

    if default_prob < 0.38:
        st.success("✅ **LOW RISK — APPROVE FULL AMOUNT:** Strong historical repayment profile.")
    elif default_prob < 0.55:
        st.warning("⚠️ **MEDIUM RISK — CONDITIONAL APPROVAL:** Consider capping the loan limit or requiring manual review.")
    else:
        st.error("🚨 **HIGH RISK — DECLINE:** High likelihood of default based on repayment velocity.")
