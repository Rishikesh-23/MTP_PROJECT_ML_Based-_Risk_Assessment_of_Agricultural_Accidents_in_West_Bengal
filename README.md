# MTP_PROJECT-(ML_Based-_Risk_Assessment_of_Agricultural_Accidents_in_West_Bengal)
This project leverages machine learning to predict the severity (fatal or non-fatal) of agricultural accidents in West Bengal. By analyzing accident data, it aims to identify risk factors, assess trends, and support preventive strategies for improving farmer safety.
---

**Description**  
This project utilizes machine learning to predict the severity (fatal or non-fatal) of agricultural accidents among farmers in West Bengal. By analyzing historical accident data, the project identifies key risk factors, trends, and insights to aid in risk assessment and develop preventive strategies for enhancing agricultural safety.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Features](#features)
4. [Dataset](#dataset)
5. [Technologies Used](#technologies-used)
6. [How to Run](#how-to-run)
7. [Results](#results)
8. [Future Scope](#future-scope)
9. [Contributors](#contributors)
10. [License](#license)

---

## **Introduction**

Agriculture is one of the most hazardous occupations, and accidents in this field can significantly impact farmers' safety and livelihoods. This project focuses on predicting the severity of agricultural accidents using machine learning models trained on accident data from West Bengal. It aims to provide actionable insights by identifying high-risk factors and visualizing accident trends, ultimately supporting the development of effective safety measures.

---

## **Project Structure**

```plaintext
Severity_Prediction_and_Risk_Assessment/
├── data/
│   ├── Final_data.csv               # Dataset containing accident records
│
├── notebooks/
│   ├── Accident_Severity_Analysis.ipynb  # Jupyter Notebook for data analysis and modeling
│
├── models/
│   ├── severity_prediction_model.pkl     # Trained ML model
│
├── app/
│   ├── app.py                           # Streamlit app for predictions
│
├── requirements.txt                     # Dependencies for the project
├── README.md                            # Project documentation
└── LICENSE                              # License file
```

---

## **Features**

- **Data Preprocessing**:
  - Cleaned and standardized accident data.
  - Removed irrelevant features and handled missing values.
- **Machine Learning Models**:
  - Trained classification models like Logistic Regression, Random Forest, and Decision Trees to predict accident severity (fatal/non-fatal).
  - Feature importance analysis to identify significant factors contributing to severity.
- **Risk Assessment**:
  - Analyzed accident trends based on location, time, and equipment usage.
  - Identified patterns that can help develop preventive strategies.
- **Interactive Application**:
  - Built a Streamlit app to allow users to input accident details and predict severity.

---

## **Dataset**

- **Source**: Data collected from accident records of West Bengal farmers.  
- **Size**: Contains detailed records of accidents, categorized by severity and associated factors.  
- **Key Features**:
  - `Year`: Year of the accident.
  - `AccidentTime`: Time of the accident.
  - `Gender`: Gender of the individual involved.
  - `EquipmentName`: Equipment involved in the accident.
  - `CauseOfAccident`: Identified cause of the accident.
  - `Reason`: Additional context or contributing factors.
  - `Severity`: Target variable indicating the accident's severity (Fatal/Non-Fatal).

---

## **Technologies Used**

- **Programming Language**: Python  
- **Libraries**:
  - Data Manipulation: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn
  - Deployment: Streamlit

---

## **How to Run**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Severity_Prediction_and_Risk_Assessment.git
   cd Severity_Prediction_and_Risk_Assessment
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (if needed):
   ```bash
   jupyter notebook notebooks/Accident_Severity_Analysis.ipynb
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

---

## **Results**

- **Model Accuracy**: Achieved 87% accuracy using the Random Forest Classifier.
- **Key Insights**:
  - Accidents involving specific equipment types had higher fatality rates.
  - Certain times of day and environmental conditions correlated with higher severity.
- **Visualizations**:
  - Graphs and charts illustrate accident trends by year, cause, and severity.

---

## **Future Scope**

1. **Data Expansion**:
   - Incorporate accident records from other regions to generalize insights.
2. **Advanced Models**:
   - Experiment with deep learning models like neural networks for better predictions.
3. **Real-Time Predictions**:
   - Integrate IoT devices to collect real-time data for immediate risk assessments.
4. **Preventive Solutions**:
   - Use insights to develop and propose safety measures specific to high-risk scenarios.

---

## **Contributors**

- **Rishikesh**  
  LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com)  
  Email: your_email@example.com  
---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
