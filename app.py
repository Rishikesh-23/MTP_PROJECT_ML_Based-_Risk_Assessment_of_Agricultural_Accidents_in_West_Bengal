import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Set up page configuration
st.set_page_config(page_title="Accident Severity Prediction", layout="wide", initial_sidebar_state="expanded")

# Header section
st.title("Accident Severity Prediction Dashboard")
st.write("An interactive tool for analyzing and predicting accident severity using various machine learning models.")

# Sidebar for data upload and feature selection
st.sidebar.header("1. Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Data successfully uploaded!")
    
    # Display Dataset
    st.write("### Data Preview")
    st.dataframe(data)

    # Preprocessing
    st.sidebar.header("2. Preprocess Data")
    if st.sidebar.button("Preprocess Data"):
        try:
            encoder = LabelEncoder()
            for column in data.select_dtypes(include=["object"]).columns:
                data[column] = encoder.fit_transform(data[column])
            st.success("Data preprocessed successfully!")
            st.write("### Data after Preprocessing")
            st.dataframe(data)
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")

    # Sidebar for model selection
    st.sidebar.header("3. Set Target Variable and Features")
    target = st.sidebar.selectbox("Select Target Variable", options=data.columns)
    features = st.sidebar.multiselect("Select Feature Variables", options=data.columns.drop(target), default=[col for col in data.columns if col != target])

    # Data Splitting
    if target and features:
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Sidebar for model selection
        st.sidebar.header("4. Choose Models to Train")
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Support Vector Machine': SVC(kernel='linear', probability=True, random_state=42)
        }
        selected_models = st.sidebar.multiselect("Select models for training:", list(models.keys()), default=['Random Forest'])

        # Train Models
        st.sidebar.header("5. Train the Model")
        if st.sidebar.button("Train Models"):
            try:
                # Validate Class Distribution
                class_counts = y.value_counts()
                if len(class_counts) < 2:
                    st.error(f"Error in model training: Only one class ({class_counts.index[0]}) is present in the selected target variable. Please choose a dataset with both classes represented.")
                else:
                    results = {}
                    for model_name in selected_models:
                        model = models[model_name]
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

                        # Collecting metrics
                        results[model_name] = {
                            'Accuracy': accuracy_score(y_test, y_pred),
                            'Precision': precision_score(y_test, y_pred, average='weighted'),
                            'Recall': recall_score(y_test, y_pred, average='weighted'),
                            'F1 Score': f1_score(y_test, y_pred, average='weighted'),
                        }

                    # Display Results
                    results_df = pd.DataFrame(results).T
                    st.write("### Model Performance")
                    st.dataframe(results_df.style.highlight_max(color='lightgreen', axis=0))

                    # Visualization
                    st.write("### Performance Metrics by Model")
                    metric = st.selectbox("Select Metric for Visualization", options=results_df.columns)
                    st.bar_chart(results_df[metric])

                    # Feature Importance for Random Forest
                    if 'Random Forest' in selected_models:
                        st.write("### Feature Importance (Random Forest)")
                        rf_model = models['Random Forest']
                        importance = rf_model.feature_importances_
                        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
                        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

                        # Plot feature importance
                        fig, ax = plt.subplots()
                        sns.barplot(x="Importance", y="Feature", data=feature_importance.head(10), palette="viridis", ax=ax)
                        st.pyplot(fig)

            except Exception as e:
                st.error(f"Error in model training: {e}")
else:
    st.sidebar.write("Awaiting CSV file upload...")
