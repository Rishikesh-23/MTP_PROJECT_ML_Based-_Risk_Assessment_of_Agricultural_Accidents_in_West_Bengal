import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set up Streamlit page configuration
st.set_page_config(page_title="Accident Severity Prediction", layout="wide", initial_sidebar_state="expanded")

# Header section
st.title("Accident Severity Prediction Dashboard")
st.write("An interactive tool for analyzing and predicting accident severity using various machine learning models.")

# Function: Preprocess Data
def preprocess_data(data):
    # Encode categorical variables
    categorical_columns = data.select_dtypes(include=['object']).columns
    encoder = LabelEncoder()
    for col in categorical_columns:
        data[col] = encoder.fit_transform(data[col])
    return data

# Sidebar for data upload
st.sidebar.header("1. Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Data uploaded successfully!")
    st.write("### Data Preview")
    st.dataframe(data.head())

    # Preprocessing step
    st.sidebar.header("2. Preprocess Data")
    if st.sidebar.button("Preprocess Data"):
        try:
            data = preprocess_data(data)
            st.success("Data preprocessing completed.")
            st.write("### Preprocessed Data")
            st.dataframe(data.head())
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")

    # Feature and Target Variable Selection
    st.sidebar.header("3. Set Target Variable and Features")
    target = st.sidebar.selectbox("Select Target Variable", options=data.columns)
    features = st.sidebar.multiselect("Select Feature Variables", options=[col for col in data.columns if col != target])

    if features and target:
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training Section
        st.sidebar.header("4. Choose Models to Train")
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Support Vector Machine': SVC(kernel='linear', probability=True)
        }
        selected_models = st.sidebar.multiselect("Select Models for Training", list(models.keys()), default=['Random Forest'])

        if st.sidebar.button("Train Models"):
            try:
                results = {}
                for model_name in selected_models:
                    model = models[model_name]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

                    results[model_name] = {
                        'Accuracy': accuracy_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred, average='weighted'),
                        'Recall': recall_score(y_test, y_pred, average='weighted'),
                        'F1 Score': f1_score(y_test, y_pred, average='weighted'),
                        'AUC': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                    }

                st.write("### Model Performance")
                st.dataframe(pd.DataFrame(results).T)

                # Visualization
                st.write("### Performance Metrics by Model")
                metric = st.selectbox("Select Metric for Visualization", options=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'])
                metric_data = {model_name: results[model_name][metric] for model_name in selected_models}
                fig, ax = plt.subplots()
                sns.barplot(x=list(metric_data.keys()), y=list(metric_data.values()), ax=ax, palette="viridis")
                ax.set_title(f"Comparison of Models on {metric}")
                ax.set_ylabel(metric)
                ax.set_xlabel("Model")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error in model training: {e}")
