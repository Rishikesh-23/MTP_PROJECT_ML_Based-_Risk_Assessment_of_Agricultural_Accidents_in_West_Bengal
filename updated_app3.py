import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Load model and label encoders.
model, le_dict = joblib.load("trained_model.pkl")

# App Config
st.set_page_config(page_title="AIS Severity Predictor", layout="wide")

# App Title
st.title("🚨 Advanced AIS Severity Prediction App")
st.markdown("Get predictions, probabilities, and expert-level insights into accident severity using ML.")

# Session State for storing prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# Define AIS Info dictionary
ais_info = {
    1: ("Minor", "Get well in primary treatment", "#4CAF50", "Non-Fatal"),
    2: ("Moderate", "Small injury", "#8BC34A", "Non-Fatal"),
    3: ("Serious", "Long term disability", "#FFC107", "Non-Fatal"),
    4: ("Severe", "Permanent damage in body parts", "#FF9800", "Non-Fatal"),
    5: ("Critical", "Exterminate of body parts", "#F44336", "Non-Fatal"),
    6: ("Unsurvivable", "Death of victims", "#D32F2F", "Fatal"),
}

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📚 AIS Info", "🆘 Help"])

# ---------------- TAB 1: PREDICTION ----------------
with tab1:
    st.subheader("Enter Accident Details to Predict AIS Level")

    col1, col2, col3 = st.columns(3)
    with col1:
        district = st.selectbox("District", le_dict["DistrictName"].classes_)
        time = st.selectbox("Time of Accident", le_dict["AccidentTime"].classes_)
        place = st.selectbox("Place of Accident", le_dict["AccidentPlace"].classes_)
    with col2:
        gender = st.selectbox("Gender", le_dict["Gender"].classes_)
        reason = st.selectbox("Reason", le_dict["Reason"].classes_)
        cause = st.selectbox("Cause of Accident", le_dict["CauseOfAccident"].classes_)
    with col3:
        cat = st.selectbox("Category", le_dict["CatDesc"].classes_)
        subcat = st.selectbox("Sub Category", le_dict["SubCatDesc"].classes_)
        equipment = st.selectbox("Equipment", le_dict["EquipmentName"].classes_)
        acc_type = st.selectbox("Accident Type", le_dict["AccidentType"].classes_)

    if st.button("🔎 Predict Severity"):
        # Prepare input
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

        # Encode
        for col in input_df.columns:
            input_df[col] = le_dict[col].transform(input_df[col])

        # Predict
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # Get severity info
        level = prediction
        severity, desc, color, category = ais_info[level]

        # Store in session history
        st.session_state.history.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "AIS Level": level,
            "Severity": severity,
            "Category": category
        })

        # Show result
        st.markdown(f"""
        <div style='background-color:{color};padding:20px;border-radius:10px'>
            <h2 style='color:white;'>AIS Level: {level}</h2>
            <h3 style='color:white;'>Severity: {severity} ({category})</h3>
            <p style='color:white;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability Chart
        st.markdown("### 🔢 AIS Level Probabilities")
        fig, ax = plt.subplots()
        labels = [f"AIS-{i}" for i in range(1, 7)]
        bars = ax.bar(labels, probabilities, color=[ais_info[i][2] for i in range(1, 7)])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        for bar, prob in zip(bars, probabilities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{prob:.2f}", ha='center', va='bottom')
        st.pyplot(fig)

        # Download report
        result_df = pd.DataFrame({
            "AIS Level": [level],
            "Severity": [severity],
            "Category": [category],
            "Description": [desc]
        })
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Result CSV", csv, "prediction_result.csv", "text/csv")

        # Top Feature Importance
        if hasattr(model, "feature_importances_"):
            st.markdown("### 🧠 Top Contributing Features")
            feat_imp = pd.Series(model.feature_importances_, index=input_df.columns)
            feat_imp = feat_imp.sort_values(ascending=False)[:5]
            st.bar_chart(feat_imp)

        # History Viewer
        with st.expander("📜 View Prediction History"):
            st.dataframe(pd.DataFrame(st.session_state.history))


# ---------------- TAB 2: AIS INFO ----------------
with tab2:
    st.subheader("📚 AIS Levels and Severity Information")
    info_df = pd.DataFrame({
        "AIS Level": ["AIS-1", "AIS-2", "AIS-3", "AIS-4", "AIS-5", "AIS-6"],
        "Classification": ["Minor", "Moderate", "Serious", "Severe", "Critical", "Unsurvivable"],
        "Description": [
            "Get well in primary treatment", "Small injury", "Long term disability",
            "Permanent damage in body parts", "Exterminate of body parts", "Death of victims"
        ],
        "Accident Category": ["Non-Fatal"] * 5 + ["Fatal"] * 1
    })
    st.table(info_df)


# ---------------- TAB 3: HELP ----------------
with tab3:
    st.subheader("🆘 How to Use This App")
    st.markdown("""
    - Go to the **Prediction** tab.
    - Enter all the accident-related details in the sidebar and hit `Predict`.
    - View the predicted **AIS Level**, **Severity**, and **Accident Category**.
    - Download the result and check class-wise probability chart.
    - All your predictions are saved in **history** for this session.

    **Note**: This app uses a trained Random Forest classifier on accident records.
    """)

    st.markdown("---")
    st.markdown("Built with ❤️ by Rishikesh")

