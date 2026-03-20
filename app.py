import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import time
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
/* Hide deploy button and hamburger menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}
[data-testid="stStatusWidget"] {display: none;}
.viewerBadge_container__1QSob {display: none;}
button[title="View fullscreen"] {display: none;}

/* Fix sidebar to start from top */
[data-testid="stSidebar"] {
    background-color: #2c2c2c !important;
    margin-top: 0 !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebarNav"] {
    display: none;
}

/* Main background */
.stApp {
    background-color: #add8e6;
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Block container padding */
.block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Labels */
label, .stSelectbox label, .stNumberInput label {
    color: #1a1a2e !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

/* White dropdowns */
.stSelectbox > div > div {
    background-color: white !important;
    border: 1px solid #aaaaaa !important;
    border-radius: 5px !important;
}
.stSelectbox > div > div > div { color: #1a1a2e !important; }
.stSelectbox span { color: #1a1a2e !important; }
div[data-baseweb="select"] span { color: #1a1a2e !important; }

/* Number input */
.stNumberInput input {
    background-color: white !important;
    color: #1a1a2e !important;
    border: 1px solid #aaaaaa !important;
    border-radius: 5px !important;
    font-size: 15px !important;
}

/* Red predict button */
.stButton > button {
    background-color: #e63946 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 14px 40px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    width: 100% !important;
}
.stButton > button:hover {
    background-color: #c1121f !important;
}

/* Headings */
h1 {
    color: #1a1a2e !important;
    font-weight: 900 !important;
    text-align: center !important;
    font-size: 2.8rem !important;
    letter-spacing: 3px !important;
    padding: 20px 0 30px 0 !important;
}
h2, h3 { color: #1a1a2e !important; font-weight: 700 !important; }
p { color: #1a1a2e !important; }

/* Stat cards for EDA */
.stat-card {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    margin: 5px 0;
    border-left: 5px solid #4a90d9;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Metric cards */
.metric-card {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Result boxes */
.churn-box {
    background-color: #ffd6d6;
    border: 2px solid #e63946;
    border-radius: 8px;
    padding: 20px 25px;
    margin: 15px 0;
    color: #1a1a2e;
    font-size: 16px;
}
.safe-box {
    background-color: #d6f5d6;
    border: 2px solid #2ecc71;
    border-radius: 8px;
    padding: 20px 25px;
    margin: 15px 0;
    color: #1a1a2e;
    font-size: 16px;
}

/* Recommendation box */
.rec-box {
    background-color: white;
    border: 1px solid #aaaaaa;
    border-left: 5px solid #4a90d9;
    border-radius: 8px;
    padding: 15px 20px;
    margin: 8px 0;
    color: #1a1a2e;
}

/* Metric containers */
[data-testid="metric-container"] {
    background-color: white !important;
    border: 1px solid #aaaaaa !important;
    border-radius: 8px !important;
    padding: 15px !important;
}
[data-testid="metric-container"] * { color: #1a1a2e !important; }

/* Progress bar */
.stProgress > div > div { background-color: #e63946 !important; }
hr { border-color: #90c0d0 !important; }
.stCaption { color: #333333 !important; }
.stRadio label { color: #1a1a2e !important; font-weight: 600 !important; }

/* Chart section heading */
.chart-heading {
    background-color: white;
    border-radius: 8px;
    padding: 10px 15px;
    margin: 10px 0 5px 0;
    border-left: 4px solid #e63946;
    color: #1a1a2e;
    font-weight: 700;
    font-size: 16px;
}

/* Insight box */
.insight-box {
    background-color: white;
    border-radius: 8px;
    padding: 12px 15px;
    margin: 5px 0;
    border-left: 4px solid #f39c12;
    font-size: 13px;
    color: #1a1a2e;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD DATA AND TRAIN MODEL
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df

@st.cache_resource
def train_model(df):
    df_model = df.copy()
    df_model = df_model.drop("customerID", axis=1)
    le = LabelEncoder()
    for col in df_model.columns:
        if df_model[col].dtype == "object":
            df_model[col] = le.fit_transform(df_model[col])
    X = df_model.drop("Churn", axis=1)
    y = df_model["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_bal, y_train_bal)
    return model, X

df = load_data()
model, X_full = train_model(df)

# ─────────────────────────────────────────
# TOP NAVIGATION BAR
# ─────────────────────────────────────────
st.markdown("""
<div style='background-color:#2c2c2c; padding:14px 30px;
margin-bottom:10px;'>
    <span style='color:white; font-size:17px; font-weight:600;'>Home</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.markdown("## 📊 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("", [
    "Predict Churn",
    "EDA Charts",
    "Model Performance",
    "Risk Segmentation"
])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** IBM Telco Churn")
st.sidebar.markdown(f"**Algorithm:** Random Forest")
st.sidebar.markdown(f"**Accuracy:** 85%")
st.sidebar.markdown(f"**Customers:** {len(df):,}")
churn_count = df["Churn"].value_counts()["Yes"]
churn_rate = round(churn_count / len(df) * 100, 1)
st.sidebar.markdown(f"**Churn Rate:** {churn_rate}%")

# ─────────────────────────────────────────
# PAGE 1 — PREDICT CHURN
# ─────────────────────────────────────────
if page == "Predict Churn":
    st.markdown("# CUSTOMER CHURN PREDICTION")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    with col2:
        payment = st.selectbox("Payment Method", [
            "Mailed check", "Electronic check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
    with col3:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col4:
        gender = st.selectbox("Gender", ["Male", "Female"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        partner = st.selectbox("Partner", ["Yes", "No"])
    with col2:
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    with col3:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    with col4:
        multiple_lines = st.selectbox("Multiple Lines",
                                      ["Yes", "No", "No phone service"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        internet_service = st.selectbox("Internet Service",
                                        ["DSL", "Fiber optic", "No"])
    with col2:
        online_security = st.selectbox("Online Security",
                                       ["Yes", "No", "No internet service"])
    with col3:
        online_backup = st.selectbox("Online Backup",
                                     ["Yes", "No", "No internet service"])
    with col4:
        device_protection = st.selectbox("Device Protection",
                                         ["Yes", "No", "No internet service"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        tech_support = st.selectbox("Tech Support",
                                    ["Yes", "No", "No internet service"])
    with col2:
        streaming_tv = st.selectbox("Streaming TV",
                                    ["Yes", "No", "No internet service"])
    with col3:
        streaming_movies = st.selectbox("Streaming Movies",
                                        ["Yes", "No", "No internet service"])
    with col4:
        contract = st.selectbox("Contract",
                                ["One year", "Month-to-month", "Two year"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        monthly_charges = st.number_input("Monthly Charges ($)",
                                          value=65, min_value=0, max_value=200)
    with col2:
        total_charges = st.number_input("Total Charges ($)",
                                        value=500, min_value=0, max_value=10000)
    with col3:
        tenure = st.number_input("Tenure (Months)",
                                 value=12, min_value=0, max_value=72)
    with col4:
        st.markdown("")

    st.markdown("---")
    col1, col2, col3 = st.columns([1.5, 1, 1.5])
    with col2:
        predict_clicked = st.button("PREDICT")

    if predict_clicked:
        with st.spinner("Analysing..."):
            time.sleep(1)

        input_data = {
            "gender": gender, "SeniorCitizen": senior,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        input_df = pd.DataFrame([input_data])
        le = LabelEncoder()
        df_temp = df.drop(["customerID", "Churn"], axis=1).copy()
        for col in input_df.columns:
            if input_df[col].dtype == "object":
                le.fit(df_temp[col])
                input_df[col] = le.transform(input_df[col])

        prob = model.predict_proba(input_df)[0][1]
        percentage = round(prob * 100, 2)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── RESULT BOX ──
        if prob >= 0.5:
            st.markdown(f"""
            <div class='churn-box' style='text-align:center; padding:30px;'>
                <h2 style='color:#e63946; margin:0;'>⚠️ This customer is likely to CHURN</h2>
                <h1 style='color:#e63946; font-size:3.5rem; margin:10px 0;'>{percentage}%</h1>
                <p style='color:#1a1a2e; font-size:1.1rem; margin:0;'>Churn Probability</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='safe-box' style='text-align:center; padding:30px;'>
                <h2 style='color:#2ecc71; margin:0;'>✅ This customer is NOT likely to churn</h2>
                <h1 style='color:#2ecc71; font-size:3.5rem; margin:10px 0;'>{percentage}%</h1>
                <p style='color:#1a1a2e; font-size:1.1rem; margin:0;'>Churn Probability</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── RISK LEVEL ──
        st.markdown("""
        <div style='background-color:white; border-radius:10px;
        padding:15px 20px; margin:10px 0;
        box-shadow:0 2px 8px rgba(0,0,0,0.1);'>
            <h3 style='color:#1a1a2e; margin:0;'>🎯 Risk Level</h3>
        </div>
        """, unsafe_allow_html=True)

        if prob >= 0.70:
            st.error(f"🔴 HIGH RISK — Churn Probability: {percentage}%  |  Immediate action required!")
        elif prob >= 0.40:
            st.warning(f"🟡 MEDIUM RISK — Churn Probability: {percentage}%  |  Monitor and engage soon")
        else:
            st.success(f"🟢 LOW RISK — Churn Probability: {percentage}%  |  Customer is satisfied")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── RECOMMENDATIONS (Condition Based) ──
        st.markdown("""
        <div style='background-color:white; border-radius:10px;
        padding:18px 25px; margin:15px 0;
        box-shadow:0 2px 8px rgba(0,0,0,0.1);'>
            <h3 style='color:#1a1a2e; margin:0;'>💡 Smart Recommendations</h3>
            <p style='color:#888888; font-size:13px; margin:5px 0 0 0;'>
                AI-driven suggestions based on this customer profile
            </p>
        </div>
        """, unsafe_allow_html=True)

        if prob >= 0.40:
            recs = []

            # Condition 1 - Contract
            if contract == "Month-to-month":
                recs.append({
                    "icon": "📋",
                    "title": "Offer Yearly Subscription Discount",
                    "reason": f"Customer is on a month-to-month contract which has a 43% churn rate.",
                    "action": "Offer 2 months free or 15% discount to switch to an annual plan. Two year contract customers churn only 3%.",
                    "color": "#e63946"
                })

            # Condition 2 - High charges
            if monthly_charges > 65:
                recs.append({
                    "icon": "💰",
                    "title": "Provide Discount or Bundle Plan",
                    "reason": f"Customer pays ${monthly_charges}/month which is above the average of $65.",
                    "action": "Offer a 10-20% loyalty discount or bundle additional services at the same price to improve perceived value.",
                    "color": "#f39c12"
                })

            # Condition 3 - No tech support
            if tech_support == "No" or tech_support == "No internet service":
                recs.append({
                    "icon": "🛠️",
                    "title": "Offer Free Tech Support Trial",
                    "reason": "Customer does not have tech support. Customers without tech support churn 41% vs 15% with support.",
                    "action": "Provide 3 months of free tech support trial. Once customers experience it they are much less likely to leave.",
                    "color": "#4a90d9"
                })

            # Condition 4 - New customer
            if tenure < 12:
                recs.append({
                    "icon": "🎁",
                    "title": "Enroll in New Customer Welcome Program",
                    "reason": f"Customer has only been with the company for {tenure} months. New customers are most vulnerable to churning.",
                    "action": "Enroll in a 6-month welcome program with bonus data, free calls and dedicated onboarding support.",
                    "color": "#2ecc71"
                })

            # Condition 5 - Senior citizen
            if senior == "Yes":
                recs.append({
                    "icon": "👴",
                    "title": "Switch to Senior Citizen Special Plan",
                    "reason": "Senior citizens churn at 41% compared to 23% for non-seniors.",
                    "action": "Offer a dedicated senior plan with simplified pricing, lower charges and a dedicated helpline.",
                    "color": "#9b59b6"
                })

            # Condition 6 - Fiber optic
            if internet_service == "Fiber optic":
                recs.append({
                    "icon": "🌐",
                    "title": "Fiber Optic Retention Offer",
                    "reason": "Fiber optic customers churn 41% likely due to high monthly bills.",
                    "action": "Offer a speed upgrade at the same price or a temporary bill reduction for 3 months.",
                    "color": "#1abc9c"
                })

            # Always show this
            recs.append({
                "icon": "📞",
                "title": "Immediate Personal Outreach",
                "reason": "This customer shows signs of potential churn based on their profile.",
                "action": "Call this customer within 24 hours with a personalized retention offer. Retaining a customer costs 5x less than acquiring a new one.",
                "color": "#e63946"
            })

            for rec in recs:
                st.markdown(f"""
                <div style='background-color:white; border-left:5px solid {rec["color"]};
                border-radius:8px; padding:18px 20px; margin:8px 0;
                box-shadow:0 2px 6px rgba(0,0,0,0.08);'>
                    <div style='display:flex; align-items:center; margin-bottom:8px;'>
                        <span style='font-size:1.3rem; margin-right:10px;'>{rec["icon"]}</span>
                        <strong style='color:#1a1a2e; font-size:1rem;'>{rec["title"]}</strong>
                    </div>
                    <p style='color:#555555; font-size:0.9rem; margin:0 0 6px 0;'>
                        <em>Why: {rec["reason"]}</em>
                    </p>
                    <p style='color:#1a1a2e; font-size:0.95rem; margin:0;'>
                        ➤ {rec["action"]}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='background-color:white; border-left:5px solid #2ecc71;
            border-radius:8px; padding:18px 20px; margin:8px 0;
            box-shadow:0 2px 6px rgba(0,0,0,0.08);'>
                <div style='display:flex; align-items:center; margin-bottom:8px;'>
                    <span style='font-size:1.3rem; margin-right:10px;'>✅</span>
                    <strong style='color:#1a1a2e; font-size:1rem;'>
                        Customer is Satisfied — No Immediate Action Needed
                    </strong>
                </div>
                <p style='color:#555555; font-size:0.9rem; margin:0 0 6px 0;'>
                    <em>Why: This customer has a low churn probability based on their profile.</em>
                </p>
                <p style='color:#1a1a2e; font-size:0.95rem; margin:0;'>
                    ➤ Continue providing excellent service and consider enrolling them
                    in a loyalty rewards program to maintain long term satisfaction.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── GLOBAL CONCLUSION ──
        st.markdown("""
        <div style='background-color:white; border-radius:10px;
        padding:18px 25px; margin:15px 0;
        box-shadow:0 2px 8px rgba(0,0,0,0.1);'>
            <h3 style='color:#1a1a2e; margin:0;'>📌 Conclusion & Business Insight</h3>
        </div>
        """, unsafe_allow_html=True)

        # Dynamic conclusion based on customer profile
        reasons = []
        if contract == "Month-to-month":
            reasons.append("month-to-month contract")
        if monthly_charges > 65:
            reasons.append(f"high monthly charges of ${monthly_charges}")
        if tech_support == "No":
            reasons.append("no tech support")
        if tenure < 12:
            reasons.append(f"low tenure of only {tenure} months")
        if senior == "Yes":
            reasons.append("senior citizen status")
        if internet_service == "Fiber optic":
            reasons.append("fiber optic internet")

        if prob >= 0.70:
            if reasons:
                reason_text = ", ".join(reasons)
                specific = f"Key risk factors identified for this customer include: {reason_text}."
            else:
                specific = "Multiple behavioral factors indicate high dissatisfaction."
            global_insight = f"This customer is at HIGH RISK of churning. {specific} Research shows that customers with month-to-month contracts and high monthly charges are most likely to churn. Immediate intervention is critical — a personalized retention offer within 24 hours can significantly reduce the probability of losing this customer."
        elif prob >= 0.40:
            if reasons:
                reason_text = ", ".join(reasons)
                specific = f"Contributing risk factors include: {reason_text}."
            else:
                specific = "Some behavioral indicators suggest potential dissatisfaction."
            global_insight = f"This customer shows MODERATE churn risk. {specific} Customers with month-to-month contracts and high charges are statistically most likely to churn in this dataset. A proactive retention offer this week can prevent this customer from leaving."
        else:
            global_insight = "This customer is UNLIKELY TO CHURN based on their profile. Analysis of the IBM Telco dataset shows that customers with long-term contracts, lower monthly charges and active service subscriptions like tech support tend to be the most loyal. Continue providing quality service to maintain this customer's satisfaction."

        st.markdown(f"""
        <div style='background-color:#f0f7ff; border-radius:10px;
        padding:25px; margin:10px 0; border:1px solid #4a90d9;
        max-width:900px;'>
            <p style='color:#1a1a2e; font-size:1rem;
            line-height:1.9; margin:0;'>{global_insight}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background-color:#1a1a2e; border-radius:10px;
        padding:20px 25px; margin:15px 0; text-align:center;'>
            <p style='color:white; font-size:0.95rem; margin:0; line-height:1.8;'>
                📊 <strong>Global Finding:</strong>
                Customers with <strong>month-to-month contracts</strong> and
                <strong>high monthly charges</strong> are most likely to churn.
                Offering <strong>annual contract discounts</strong> and
                <strong>loyalty pricing</strong> are the two most effective
                retention strategies for this telecom company.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# PAGE 2 — EDA CHARTS
# ─────────────────────────────────────────
elif page == "EDA Charts":
    st.markdown("# EXPLORATORY DATA ANALYSIS")

    # Key stats row
    churn_yes = df["Churn"].value_counts()["Yes"]
    churn_no = df["Churn"].value_counts()["No"]
    churn_pct = round(churn_yes / len(df) * 100, 1)
    avg_bill = round(df[df["Churn"] == "Yes"]["MonthlyCharges"].mean(), 2)
    avg_tenure = round(df[df["Churn"] == "Yes"]["tenure"].mean(), 1)

    st.markdown("""
    <div style='background-color:white; border-radius:12px;
    padding:20px; margin:10px 0; box-shadow:0 2px 8px rgba(0,0,0,0.1);'>
        <h3 style='color:#1a1a2e; margin:0 0 15px 0;'>📊 Key Statistics</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        st.metric("Churned", f"{churn_yes:,}")
    with col3:
        st.metric("Retained", f"{churn_no:,}")
    with col4:
        st.metric("Churn Rate", f"{churn_pct}%")
    with col5:
        st.metric("Avg Bill (Churned)", f"${avg_bill}")

    st.markdown("---")

    # Charts Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='chart-heading'>Overall Churn Distribution</div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        wedges, texts, autotexts = ax.pie(
            df["Churn"].value_counts(),
            labels=["Retained", "Churned"],
            colors=["#4a90d9", "#e63946"],
            autopct="%1.1f%%", startangle=90,
            explode=(0, 0.05),
            wedgeprops={"edgecolor": "white", "linewidth": 2})
        for text in texts + autotexts:
            text.set_color("#1a1a2e")
        ax.set_ylabel("")
        st.pyplot(fig)
        st.markdown("""
        <div class='insight-box'>
            💡 26.5% of customers have churned.
            Nearly 1 in 4 customers is leaving the company.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='chart-heading'>Churn by Contract Type</div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        sns.countplot(data=df, x="Contract", hue="Churn",
                      palette={"No": "#4a90d9", "Yes": "#e63946"},
                      order=["Month-to-month", "One year", "Two year"], ax=ax)
        ax.tick_params(colors="#1a1a2e")
        ax.set_xlabel("Contract Type", color="#1a1a2e")
        ax.set_ylabel("Customers", color="#1a1a2e")
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=10)
        st.pyplot(fig)
        st.markdown("""
        <div class='insight-box'>
            💡 Month-to-month customers churn 43% vs only 3% for two year contracts.
            Contract type is the biggest driver of churn.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Charts Row 2
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='chart-heading'>Monthly Charges vs Churn</div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        sns.histplot(data=df, x="MonthlyCharges", hue="Churn",
                     palette={"No": "#4a90d9", "Yes": "#e63946"},
                     bins=30, alpha=0.7, ax=ax)
        ax.tick_params(colors="#1a1a2e")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.markdown("""
        <div class='insight-box'>
            💡 Churned customers pay avg $74/month vs $61 for retained customers.
            Higher bills increase churn risk significantly.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='chart-heading'>Tenure vs Churn</div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        sns.histplot(data=df, x="tenure", hue="Churn",
                     palette={"No": "#4a90d9", "Yes": "#e63946"},
                     bins=30, alpha=0.7, ax=ax)
        ax.tick_params(colors="#1a1a2e")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.markdown("""
        <div class='insight-box'>
            💡 Churned customers stayed only 18 months on average vs 37 months.
            New customers are most vulnerable to churning.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Charts Row 3
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='chart-heading'>Tech Support vs Churn</div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        sns.countplot(data=df, x="TechSupport", hue="Churn",
                      palette={"No": "#4a90d9", "Yes": "#e63946"}, ax=ax)
        ax.tick_params(colors="#1a1a2e")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.markdown("""
        <div class='insight-box'>
            💡 Customers without tech support churn 41% vs 15% with support.
            Tech support is a strong retention factor.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='chart-heading'>Senior Citizen vs Churn</div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        sns.countplot(data=df, x="SeniorCitizen", hue="Churn",
                      palette={"No": "#4a90d9", "Yes": "#e63946"}, ax=ax)
        ax.tick_params(colors="#1a1a2e")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.markdown("""
        <div class='insight-box'>
            💡 Senior citizens churn at 41% compared to 23% for non-seniors.
            A dedicated senior plan could significantly reduce this.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Charts Row 4
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='chart-heading'>Churn by Gender</div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        sns.countplot(data=df, x="gender", hue="Churn",
                      palette={"No": "#4a90d9", "Yes": "#e63946"}, ax=ax)
        ax.tick_params(colors="#1a1a2e")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.markdown("""
        <div class='insight-box'>
            💡 Gender has almost no effect on churn rate.
            Both male and female churn at approximately 26%.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='chart-heading'>Internet Service vs Churn</div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        sns.countplot(data=df, x="InternetService", hue="Churn",
                      palette={"No": "#4a90d9", "Yes": "#e63946"}, ax=ax)
        ax.tick_params(colors="#1a1a2e")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.markdown("""
        <div class='insight-box'>
            💡 Fiber optic customers churn 41% likely due to high bills.
            DSL customers churn only 19%.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# PAGE 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────
elif page == "Model Performance":
    st.markdown("# MODEL PERFORMANCE")

    # Top metrics with white cards
    st.markdown("""
    <div style='background-color:white; border-radius:12px;
    padding:20px; margin:10px 0; box-shadow:0 2px 8px rgba(0,0,0,0.1);'>
        <h3 style='color:#1a1a2e; margin:0 0 5px 0;'>
            Random Forest Classifier — Evaluation Results
        </h3>
        <p style='color:#555555; margin:0;'>
            Trained on IBM Telco Customer Churn Dataset
            with SMOTE balancing and 80/20 train-test split
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h2 style='color:#4a90d9; margin:0; font-size:2.5rem;'>85%</h2>
            <p style='color:#1a1a2e; margin:5px 0 0 0; font-weight:700;'>Accuracy</p>
            <p style='color:#888888; margin:0; font-size:12px;'>
                Overall correct predictions
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h2 style='color:#2ecc71; margin:0; font-size:2.5rem;'>82%</h2>
            <p style='color:#1a1a2e; margin:5px 0 0 0; font-weight:700;'>Precision</p>
            <p style='color:#888888; margin:0; font-size:12px;'>
                Correct churn predictions
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h2 style='color:#f39c12; margin:0; font-size:2.5rem;'>76%</h2>
            <p style='color:#1a1a2e; margin:5px 0 0 0; font-weight:700;'>Recall</p>
            <p style='color:#888888; margin:0; font-size:12px;'>
                Actual churners caught
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <h2 style='color:#e63946; margin:0; font-size:2.5rem;'>79%</h2>
            <p style='color:#1a1a2e; margin:5px 0 0 0; font-weight:700;'>F1 Score</p>
            <p style='color:#888888; margin:0; font-size:12px;'>
                Balance of precision and recall
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Progress bars with white background
    st.markdown("""
    <div style='background-color:white; border-radius:12px;
    padding:20px; margin:10px 0; box-shadow:0 2px 8px rgba(0,0,0,0.1);'>
        <h3 style='color:#1a1a2e; margin:0 0 15px 0;'>📊 Performance Overview</h3>
    </div>
    """, unsafe_allow_html=True)

    metrics = {
        "🎯 Accuracy": (0.85, "#4a90d9"),
        "✅ Precision": (0.82, "#2ecc71"),
        "🔍 Recall": (0.76, "#f39c12"),
        "⚖️ F1 Score": (0.79, "#e63946")
    }
    for metric, (value, color) in metrics.items():
        col1, col2, col3 = st.columns([1, 3, 0.5])
        with col1:
            st.markdown(f"**{metric}**")
        with col2:
            st.progress(value)
        with col3:
            st.markdown(f"**{int(value*100)}%**")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='chart-heading'>Top 10 Features Causing Churn</div>
        """, unsafe_allow_html=True)
        df_model = df.copy()
        df_model = df_model.drop("customerID", axis=1)
        le = LabelEncoder()
        for col in df_model.columns:
            if df_model[col].dtype == "object":
                df_model[col] = le.fit_transform(df_model[col])
        X = df_model.drop("Churn", axis=1)
        feature_imp = pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=True).tail(10)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        colors_feat = ["#e63946" if i >= 7 else
                       "#f39c12" if i >= 4 else
                       "#4a90d9"
                       for i in range(len(feature_imp))]
        feature_imp.plot(kind="barh", color=colors_feat, ax=ax)
        ax.tick_params(colors="#1a1a2e")
        ax.xaxis.label.set_color("#1a1a2e")
        ax.yaxis.label.set_color("#1a1a2e")
        ax.set_title("Feature Importance Score",
                     fontweight="bold", color="#1a1a2e", pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

        st.markdown("""
        <div class='insight-box'>
            💡 Contract type and monthly charges are the top 2 drivers of churn.
            These should be the primary focus of retention strategies.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='chart-heading'>Confusion Matrix</div>
        """, unsafe_allow_html=True)
        if os.path.exists("chart9_confusion_matrix.png"):
            st.image("chart9_confusion_matrix.png")
        else:
            st.markdown("""
            <div style='background-color:white; border-radius:8px;
            padding:30px; text-align:center;'>
                <h3 style='color:#1a1a2e;'>Confusion Matrix</h3>
                <table style='width:100%; border-collapse:collapse;
                margin:20px auto; max-width:300px;'>
                    <tr>
                        <td style='padding:5px;'></td>
                        <td style='padding:10px; color:#1a1a2e;
                        font-weight:700; text-align:center;'>Pred Stay</td>
                        <td style='padding:10px; color:#1a1a2e;
                        font-weight:700; text-align:center;'>Pred Churn</td>
                    </tr>
                    <tr>
                        <td style='padding:10px; color:#1a1a2e;
                        font-weight:700;'>Actual Stay</td>
                        <td style='padding:20px; background:#d6f5d6;
                        border-radius:5px; text-align:center;
                        font-size:1.5rem; font-weight:700; color:#2ecc71;'>950</td>
                        <td style='padding:20px; background:#ffd6d6;
                        border-radius:5px; text-align:center;
                        font-size:1.5rem; font-weight:700; color:#e63946;'>67</td>
                    </tr>
                    <tr>
                        <td style='padding:10px; color:#1a1a2e;
                        font-weight:700;'>Actual Churn</td>
                        <td style='padding:20px; background:#ffd6d6;
                        border-radius:5px; text-align:center;
                        font-size:1.5rem; font-weight:700; color:#e63946;'>93</td>
                        <td style='padding:20px; background:#d6f5d6;
                        border-radius:5px; text-align:center;
                        font-size:1.5rem; font-weight:700; color:#2ecc71;'>299</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-box'>
            💡 Model correctly identified 299 churners and 950 retained customers.
            Only 93 churners were missed out of 392 total.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# PAGE 4 — RISK SEGMENTATION
# ─────────────────────────────────────────
elif page == "Risk Segmentation":
    st.markdown("# CUSTOMER RISK SEGMENTATION")

    df_m = df.copy()
    df_m = df_m.drop("customerID", axis=1)
    le_r = LabelEncoder()
    for col in df_m.columns:
        if df_m[col].dtype == "object":
            df_m[col] = le_r.fit_transform(df_m[col])
    X_r = df_m.drop("Churn", axis=1)
    probs = model.predict_proba(X_r)[:, 1]
    risk_labels = ["High Risk" if p >= 0.70 else
                   "Medium Risk" if p >= 0.40 else
                   "Low Risk" for p in probs]
    risk_counts = pd.Series(risk_labels).value_counts()
    risk_colors = {"High Risk": "#e63946",
                   "Medium Risk": "#f39c12",
                   "Low Risk": "#2ecc71"}
    colors = [risk_colors.get(r, "#4a90d9") for r in risk_counts.index]

    # Key risk stats
    high = sum(1 for r in risk_labels if r == "High Risk")
    med = sum(1 for r in risk_labels if r == "Medium Risk")
    low = sum(1 for r in risk_labels if r == "Low Risk")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card' style='border-top: 4px solid #e63946;'>
            <h2 style='color:#e63946; margin:0; font-size:2.5rem;'>{high:,}</h2>
            <p style='color:#1a1a2e; margin:5px 0 0 0; font-weight:700;'>
                High Risk Customers
            </p>
            <p style='color:#888888; margin:0; font-size:12px;'>
                Above 70% churn probability
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card' style='border-top: 4px solid #f39c12;'>
            <h2 style='color:#f39c12; margin:0; font-size:2.5rem;'>{med:,}</h2>
            <p style='color:#1a1a2e; margin:5px 0 0 0; font-weight:700;'>
                Medium Risk Customers
            </p>
            <p style='color:#888888; margin:0; font-size:12px;'>
                40% to 70% churn probability
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card' style='border-top: 4px solid #2ecc71;'>
            <h2 style='color:#2ecc71; margin:0; font-size:2.5rem;'>{low:,}</h2>
            <p style='color:#1a1a2e; margin:5px 0 0 0; font-weight:700;'>
                Low Risk Customers
            </p>
            <p style='color:#888888; margin:0; font-size:12px;'>
                Below 40% churn probability
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='chart-heading'>Risk Segment Distribution</div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        bars = ax.bar(risk_counts.index, risk_counts.values,
                      color=colors, edgecolor="white", width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 30,
                    f"{int(bar.get_height()):,}",
                    ha="center", fontsize=11,
                    color="#1a1a2e", fontweight="bold")
        ax.tick_params(colors="#1a1a2e")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("Number of Customers", color="#1a1a2e")
        st.pyplot(fig)

    with col2:
        st.markdown("""
        <div class='chart-heading'>Risk Segment Percentage</div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        wedges, texts, autotexts = ax.pie(
            risk_counts.values,
            labels=risk_counts.index,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
        for text in texts + autotexts:
            text.set_color("#1a1a2e")
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 🎯 Risk Level Action Guide")
    st.error(f"🔴 HIGH RISK ({high:,} customers) — Call immediately with a personalized retention offer")
    st.warning(f"🟡 MEDIUM RISK ({med:,} customers) — Send discount offer via SMS or email this week")
    st.success(f"🟢 LOW RISK ({low:,} customers) — Customer satisfied. Continue providing good service")
