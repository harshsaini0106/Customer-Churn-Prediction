import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load model
# -----------------------------
model = pickle.load(open("churn_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

# -----------------------------
# CUSTOM CSS (Refined UI)
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
}

.stButton>button {
    background: linear-gradient(90deg, #00C9A7, #00B4DB);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: 600;
}

.small-text {
    color: #94a3b8;
    font-size: 14px;
}

.big-text {
    font-size: 32px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("🚀 Customer Churn Prediction Dashboard")
st.markdown('<p class="small-text">Predict customer churn using Machine Learning</p>', unsafe_allow_html=True)

st.divider()

# -----------------------------
# INPUT SECTION
# -----------------------------
col1, col2 = st.columns([1,2])

with col1:
    st.markdown("## 🧾 Customer Details")

    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

    contract = st.selectbox("Contract Type",
        ["Month-to-month", "One year", "Two year"])

    internet = st.selectbox("Internet Service",
        ["DSL", "Fiber optic", "No"])

    payment = st.selectbox("Payment Method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"])

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
total_charges = tenure * monthly_charges

input_dict = {col: 0 for col in model_columns}

input_dict["tenure"] = tenure
input_dict["MonthlyCharges"] = monthly_charges
input_dict["TotalCharges"] = total_charges

input_dict[f"Contract_{contract}"] = 1
input_dict[f"InternetService_{internet}"] = 1
input_dict[f"PaymentMethod_{payment}"] = 1

df = pd.DataFrame([input_dict])
df = df[model_columns]
df_scaled = scaler.transform(df)

# -----------------------------
# PREDICTION SECTION
# -----------------------------
with col2:
    st.markdown("## 📊 Prediction Results")

    if st.button("🔍 Predict Now"):

        prediction = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0][1]

        # KPI Cards
        kpi1, kpi2 = st.columns(2)

        with kpi1:
            st.markdown(f"""
            <div class="card">
                <div class="small-text">Churn Probability</div>
                <div class="big-text">{prob:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            status = "Churn" if prediction == 1 else "Stay"
            st.markdown(f"""
            <div class="card">
                <div class="small-text">Status</div>
                <div class="big-text">{status}</div>
            </div>
            """, unsafe_allow_html=True)

        # Result Message
        if prediction == 1:
            st.error("⚠️ High Risk Customer")
        else:
            st.success("✅ Customer Likely to Stay")

        st.divider()

        # -----------------------------
        # CLEAN SMALL CHART
        # -----------------------------
        st.markdown("### 📈 Prediction Chart")

        col_center1, col_center2, col_center3 = st.columns([1,2,1])

        with col_center2:
            fig, ax = plt.subplots(figsize=(3.5,2.5))
            ax.bar(["Stay", "Churn"], [1-prob, prob])

            ax.spines[['top','right']].set_visible(False)
            ax.set_ylabel("")
            ax.set_title("")

            st.pyplot(fig, use_container_width=False)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📌 About Project")

st.sidebar.markdown("""
### 💡 Features
- Machine Learning Model  
- Real-time Prediction  
- Feature Engineering  
- SMOTE for imbalance  

### ⚙️ Tech Stack
- Python  
- Scikit-learn  
- Streamlit  

### 🎯 Goal
Predict customer churn and help businesses retain customers.
""")