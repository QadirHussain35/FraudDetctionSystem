import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Load the dataset
df = pd.read_csv("Credit_Card_Fraud.csv")

# Data Preprocessing
print(df.info())
print(df.describe())
print(df.head())

# Drop unnecessary columns
df.drop(['Unnamed: 0', 'first', 'last', 'street', 'city', 'zip', 'trans_num'], axis=1, inplace=True)

# Convert date columns
df['dob'] = pd.to_datetime(df['dob'], dayfirst=True)
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%d/%m/%Y %H:%M')

# Calculate age
df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year
df['age'] -= ((df['trans_date_trans_time'].dt.month < df['dob'].dt.month) | 
              ((df['trans_date_trans_time'].dt.month == df['dob'].dt.month) & 
               (df['trans_date_trans_time'].dt.day < df['dob'].dt.day))).astype(int)

print(df['age'].head())
print(df.isnull().sum())

# Sample 10,000 rows for faster processing
df_sample = df.sample(n=10000, random_state=42)

# Convert categorical variables to dummy variables
df_sample = pd.get_dummies(df_sample, columns=['merchant', 'category', 'gender', 'job', 'state'], drop_first=True)

# Normalize numerical columns
scaler = StandardScaler()
numerical_cols = ['amt', 'lat', 'long', 'city_pop', 'age', 'merch_lat', 'merch_long']
df_sample[numerical_cols] = scaler.fit_transform(df_sample[numerical_cols])

# Prepare features and target variable
X = df_sample.drop(columns=['is_fraud', 'dob', 'trans_date_trans_time'])
y = df_sample['is_fraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance in the training set
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train the Random Forest model with class_weight='balanced'
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=1)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=1)
grid_search.fit(X_train, y_train)

# Display the best parameters and evaluate the best model
print(f"Best parameters found: {grid_search.best_params_}")
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))

# Feature importance
importances = best_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df.head())

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(np.arange(2), ['Not Fraud', 'Fraud'])
plt.yticks(np.arange(2), ['Not Fraud', 'Fraud'])
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_best)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Save the model
joblib.dump(best_model, 'fraud_detection_model.pkl')
loaded_model = joblib.load('fraud_detection_model.pkl')

# Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        expected_features = ['amt', 'lat', 'long', 'city_pop', 'age', 'merch_lat', 'merch_long', 'merchant_X', 'category_X', 'gender_X', 'job_X', 'state_X']
        if not all(feature in data for feature in expected_features):
            return jsonify({'error': 'Missing required features in input data'}), 400
        features = np.array([list(data.values())]).reshape(1, -1)
        prediction = model.predict(features)
        result = {'prediction': int(prediction[0])}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
