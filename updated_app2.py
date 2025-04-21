import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load model and encoders
model, le_dict = joblib.load("trained_model.pkl")

# AIS Information
ais_info = {
    1: ("Moderate", "Get well in primary treatment", "#4CAF50", "Non-Fatal", "🙂"),
    2: ("Moderate", "Small injury", "#8BC34A", "Non-Fatal", "🙂"),
    3: ("Serious", "Long term disability", "#FFC107", "Non-Fatal", "😐"),
    4: ("Serious", "Permanent damage in body parts", "#FF9800", "Non-Fatal", "😟"),
    5: ("Severe", "Exterminate of body parts", "#F44336", "Fatal", "😨"),
    6: ("Severe", "Death of victims", "#D32F2F", "Fatal", "💀"),
}

# Streamlit Config
st.set_page_config(page_title="🚨 AIS Predictor", layout="wide")

# Title Section
st.title("🚨 AIS Injury Severity Prediction")
st.markdown("Predict the **AIS level** and get insights based on accident-related features.")

# Form Input Grouping
with st.sidebar:
    st.header("📝 Accident Info")
    district = st.selectbox("District", le_dict["DistrictName"].classes_)
    time = st.selectbox("Time of Accident", le_dict["AccidentTime"].classes_)
    place = st.selectbox("Place", le_dict["AccidentPlace"].classes_)
    reason = st.selectbox("Reason", le_dict["Reason"].classes_)
    cause = st.selectbox("Cause", le_dict["CauseOfAccident"].classes_)

    st.header("🛠️ Equipment Info")
    cat = st.selectbox("Category", le_dict["CatDesc"].classes_)
    subcat = st.selectbox("SubCategory", le_dict["SubCatDesc"].classes_)
    equipment = st.selectbox("Equipment", le_dict["EquipmentName"].classes_)

    st.header("👤 Personal Info")
    gender = st.radio("Gender", le_dict["Gender"].classes_)
    acc_type = st.selectbox("Accident Type", le_dict["AccidentType"].classes_)

# Prepare input for prediction
input_data = {
    "DistrictName": district,
    "AccidentTime": time,
    "AccidentPlace": place,
    "Gender": gender,
    "CatDesc": cat,
    "SubCatDesc": subcat,
    "EquipmentName": equipment,
    "CauseOfAccident": cause,
    "Reason": reason,
    "AccidentType": acc_type,
}

input_df = pd.DataFrame([input_data])
for col in input_df.columns:
    input_df[col] = le_dict[col].transform(input_df[col])

# Predict
prediction = model.predict(input_df)[0]
probabilities = model.predict_proba(input_df)[0]

# Extract Details
severity, description, color, category, emoji = ais_info[prediction]

# 🎯 Display Results
st.markdown("### 🧾 Prediction Result")
st.markdown(f"""
<div style="background-color:{color};padding:20px;border-radius:10px;text-align:center">
    <h2 style="color:white;">{emoji} AIS Level: {prediction}</h2>
    <h3 style="color:white;">Severity: {severity} ({category})</h3>
    <p style="color:white;">{description}</p>
</div>
""", unsafe_allow_html=True)

# 📊 Show Probability for All Classes
st.markdown("### 📊 Class Probabilities")
labels = [f"AIS-{i}" for i in range(1, 7)]
fig, ax = plt.subplots()
bars = ax.bar(labels, probabilities, color=['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336', '#D32F2F'])
ax.set_ylim(0, 1)
ax.set_ylabel("Probability")
for bar, prob in zip(bars, probabilities):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{prob:.2f}", ha='center', va='bottom')
st.pyplot(fig)

# ⬇️ Download prediction report
st.markdown("### 📁 Download Report")
report_df = pd.DataFrame({
    "AIS Level": [f"AIS-{prediction}"],
    "Classification": [severity],
    "Category": [category],
    "Description": [description],
})
csv = report_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "ais_prediction.csv", "text/csv")

# 🧠 Show Feature Importances (if available)
st.markdown("### 🧠 Top Contributing Features")
if hasattr(model, 'feature_importances_'):
    feat_imp = pd.Series(model.feature_importances_, index=input_df.columns)
    feat_imp = feat_imp.sort_values(ascending=False)[:5]
    st.bar_chart(feat_imp)

# 📋 AIS Severity Scale
st.markdown("---")
st.markdown("### 📝 AIS Scale Reference Table")
ais_df = pd.DataFrame({
    "AIS Level": ["AIS-1", "AIS-2", "AIS-3", "AIS-4", "AIS-5", "AIS-6"],
    "Classification": ["Moderate", "Moderate", "Serious", "Serious", "Severe", "Severe"],
    "Description": [
        "Get well in primary treatment", "Small injury", "Long term disability",
        "Permanent damage in body parts", "Exterminate of body parts", "Death of victims"
    ],
    "Accident Category": ["Non-Fatal"] * 4 + ["Fatal"] * 2
})
st.dataframe(ais_df)
