import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoders
model, le_dict = joblib.load('trained_model.pkl')

# AIS severity details
ais_info = {
    1: ('Moderate', 'Get well in primary treatment', '#4CAF50'),  # Green
    2: ('Moderate', 'Small injury', '#8BC34A'),  # Light green
    3: ('Serious', 'Long term disability', '#FFC107'),  # Amber
    4: ('Serious', 'Permanent damage in body parts', '#FF9800'),  # Orange
    5: ('Severe', 'Exterminate of body parts', '#F44336'),  # Light Red
    6: ('Severe', 'Death of victims', '#D32F2F')  # Dark Red
}

st.set_page_config(page_title="AIS Severity Prediction App", page_icon="🚨", layout="wide")

# App title and description
st.title("🚨 AIS Severity Level Prediction")
st.markdown("""
Predict and visualize **Accident Severity** based on user selections.
""")

# Sidebar for inputs
st.sidebar.header("Enter Accident Details")

def user_input_features():
    inputs = {}
    inputs['DistrictName'] = st.sidebar.selectbox('District Name', le_dict['DistrictName'].classes_)
    inputs['AccidentTime'] = st.sidebar.selectbox('Accident Time', le_dict['AccidentTime'].classes_)
    inputs['AccidentPlace'] = st.sidebar.selectbox('Accident Place', le_dict['AccidentPlace'].classes_)
    inputs['Gender'] = st.sidebar.selectbox('Gender', le_dict['Gender'].classes_)
    inputs['CatDesc'] = st.sidebar.selectbox('Category Description', le_dict['CatDesc'].classes_)
    inputs['SubCatDesc'] = st.sidebar.selectbox('Sub-Category Description', le_dict['SubCatDesc'].classes_)
    inputs['EquipmentName'] = st.sidebar.selectbox('Equipment Name', le_dict['EquipmentName'].classes_)
    inputs['CauseOfAccident'] = st.sidebar.selectbox('Cause of Accident', le_dict['CauseOfAccident'].classes_)
    inputs['Reason'] = st.sidebar.selectbox('Reason of Accident', le_dict['Reason'].classes_)
    inputs['AccidentType'] = st.sidebar.selectbox('Accident Type', le_dict['AccidentType'].classes_)
    
    return inputs

user_data = user_input_features()
user_df = pd.DataFrame([user_data])

# Encoding input
for col in user_df.columns:
    le = le_dict[col]
    user_df[col] = le.transform(user_df[col])

# Prediction
prediction = model.predict(user_df)[0]

# Displaying Results
severity, description, color = ais_info[prediction]

st.markdown(f"""
<div style="background-color:{color};padding:15px;border-radius:10px;text-align:center">
    <h2 style="color:white;">AIS Level: {prediction}</h2>
    <h3 style="color:white;">Classification: {severity}</h3>
    <p style="color:white;">{description}</p>
</div>
""", unsafe_allow_html=True)

# Display AIS details table
st.markdown("---")
st.subheader("📝 AIS Severity Scale")
ais_df = pd.DataFrame({
    "AIS Level": ["AIS-1", "AIS-2", "AIS-3", "AIS-4", "AIS-5", "AIS-6"],
    "Classification": ["Moderate", "Moderate", "Serious", "Serious", "Severe", "Severe"],
    "Description": [
        "Get well in primary treatment", "Small injury", "Long term disability",
        "Permanent damage in body parts", "Exterminate of body parts", "Death of victims"
    ]
})

def color_severity(val):
    colors = {
        "Moderate": "#8BC34A",
        "Serious": "#FFC107",
        "Severe": "#F44336"
    }
    return f'background-color: {colors[val]}'

st.table(ais_df.style.applymap(color_severity, subset=['Classification']))
