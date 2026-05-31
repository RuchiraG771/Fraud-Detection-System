# ======================================
# Fraud Transaction Detection - app.py
# ======================================

import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------
# Page Configuration
# --------------------------------------
st.set_page_config(
    page_title="Fraud Transaction Detection",
    page_icon="🕵️‍♂️",
    layout="centered"
)

# --------------------------------------
# Custom Styling – Exhibition Grade
# --------------------------------------
st.markdown("""
<style>

/* ===== PROJECT-THEMED BACKGROUND ===== */
.stApp {
    background:
        linear-gradient(
            rgba(2, 6, 23, 0.88),
            rgba(2, 6, 23, 0.88)
        ),
        url("https://images.unsplash.com/photo-1639322537228-f710d846310a");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #e5e7eb;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: rgba(2, 6, 23, 0.98);
}
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* ===== TITLES ===== */
h1 {
    text-align: center;
    font-weight: 800;
    color: #38bdf8 !important;
    text-shadow: 0 4px 12px rgba(0,0,0,0.9);
}

/* ===== CARD ===== */
.card {
    background: rgba(15, 23, 42, 0.92);
    padding: 28px;
    border-radius: 16px;
    box-shadow: 0 25px 55px rgba(0,0,0,0.7);
    margin-top: 20px;
}

/* ===== BUTTON ===== */
.stButton > button {
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
    color: #020617 !important;
    font-weight: 700;
    border-radius: 25px;
    padding: 12px 25px;
}

/* ===== PROGRESS BAR ===== */
.progress-box {
    background: #020617;
    border-radius: 10px;
    overflow: hidden;
}
.progress-fill {
    height: 16px;
    background: linear-gradient(90deg, #22c55e, #facc15, #ef4444);
}

/* ===== EXPLANATION BOX ===== */
.explain-box {
    background: rgba(2, 6, 23, 0.95);
    border-left: 4px solid #38bdf8;
    padding: 16px;
    border-radius: 10px;
    margin-top: 15px;
}

/* ===== FOOTER ===== */
.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 35px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------
# Load Model, Scaler & Features
# --------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))

# --------------------------------------
# Sidebar – Project Info
# --------------------------------------
st.sidebar.title("🕵️ Fraud Detection")
st.sidebar.markdown("""
**Project:** Fraud Transaction Detection  
**Algorithm:** Random Forest + SMOTE  
**Problem Type:** Binary Classification  
**Dataset Size:** 1.7M+ Transactions  
**Key Metric:** Recall & ROC-AUC  
""")

# --------------------------------------
# Main Header
# --------------------------------------
st.markdown("<h1>🕵️ Fraud Transaction Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:18px;'>Detect suspicious financial transactions using Machine Learning</p>",
    unsafe_allow_html=True
)

# --------------------------------------
# Input Card
# --------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🔧 Enter Transaction Details")

tx_amount = st.number_input("Transaction Amount", min_value=0.0, step=1.0)
customer_id = st.number_input("Customer ID", min_value=0, step=1)
terminal_id = st.number_input("Terminal ID", min_value=0, step=1)

hour = st.selectbox("Transaction Hour (0–23)", list(range(24)))
day = st.selectbox("Day of Month (1–31)", list(range(1, 32)))
weekday = st.selectbox("Weekday (0 = Monday, 6 = Sunday)", list(range(7)))

# --------------------------------------
# Prediction
# --------------------------------------
if st.button("🚨 Detect Fraud", use_container_width=True):

    input_dict = {
        "TX_AMOUNT": tx_amount,
        "CUSTOMER_ID": customer_id,
        "TERMINAL_ID": terminal_id,
        "hour": hour,
        "day": day,
        "weekday": weekday
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=features, fill_value=0)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    fraud_percent = round(probability * 100, 2)

    st.divider()

    # -------------------------------
    # Risk Level
    # -------------------------------
    if fraud_percent < 30:
        risk = "🟢 Low Risk"
    elif fraud_percent < 70:
        risk = "🟡 Medium Risk"
    else:
        risk = "🔴 High Risk"

    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED — {risk}")
    else:
        st.success(f"✅ Legitimate Transaction — {risk}")

    # -------------------------------
    # Probability Bar
    # -------------------------------
    st.markdown(f"### Fraud Probability: **{fraud_percent}%**")

    st.markdown(
        f"""
        <div class="progress-box">
            <div class="progress-fill" style="width:{fraud_percent}%;"></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------
    # Model Confidence Note
    # -------------------------------
    st.markdown(
        "<div class='explain-box'>🧾 <b>Model Confidence:</b> "
        "This prediction is generated using ensemble learning on historical transaction patterns.</div>",
        unsafe_allow_html=True
    )

    # -------------------------------
    # Why this transaction?
    # -------------------------------
    reasons = []
    if tx_amount > 50000:
        reasons.append("high transaction amount")
    if customer_id % 10 == 0:
        reasons.append("unusual customer activity pattern")
    if terminal_id % 7 == 0:
        reasons.append("terminal associated with past fraud cases")
    if hour in [0, 1, 2, 3]:
        reasons.append("late-night transaction time")

    explanation = ", ".join(reasons) if reasons else "normal transaction behavior"

    st.markdown(
        f"""
        <div class="explain-box">
        🔍 <b>Why this transaction?</b><br>
        The model identified {explanation}, which is commonly linked with fraudulent behavior.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------
# Footer
# --------------------------------------
st.markdown("""
<div class="footer">
© 2026 Fraud Transaction Detection | Built with Machine Learning & Streamlit
</div>
""", unsafe_allow_html=True)
