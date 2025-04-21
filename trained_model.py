import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv("C:\\Users\\02ris\\OneDrive\\Desktop\\mtp project related pdf\\Data\\Final_data.csv")

data['SubCatDesc'].fillna('Unknown', inplace=True)
data['Reason'].fillna('Unknown', inplace=True)

features = ['DistrictName', 'AccidentTime', 'AccidentPlace', 'Gender', 'CatDesc',
            'SubCatDesc', 'EquipmentName', 'CauseOfAccident', 'Reason', 'AccidentType']
target = 'AIS level'

le_dict = {}
for col in features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

joblib.dump((model, le_dict), 'trained_model.pkl')
