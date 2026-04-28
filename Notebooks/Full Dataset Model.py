# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 22:19:16 2026

@author: Hp
"""

# ==========================================
# 📌 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')


# ==========================================
# 📌 2. LOAD DATA
# ==========================================
data= pd.read_csv(r"C:\Users\Hp\Videos\Python,ML & DL\Insuarnce Claims\insurance_claims.csv")

print(data)

# Replace ? with NaN
data.replace('?', np.nan, inplace=True)


# ==========================================
# 📌 3. HANDLE MISSING VALUES
# ==========================================
data['collision_type'].fillna('unknown', inplace=True)
data['authorities_contacted'].fillna('unknown', inplace=True)
data['property_damage'].fillna('unknown', inplace=True)
data['police_report_available'].fillna('unknown', inplace=True)


# ==========================================
# 📌 4. DATA CLEANING
# ==========================================
data.drop('_c39', axis=1, inplace=True)
data.drop_duplicates(inplace=True)

# Convert dates
data['policy_bind_date'] = pd.to_datetime(data['policy_bind_date'])
data['incident_date'] = pd.to_datetime(data['incident_date'])

# Feature engineering
data['policy_age_days'] = (data['incident_date'] - data['policy_bind_date']).dt.days


# ==========================================
# 📌 5. DROP UNUSED COLUMNS
# ==========================================
drop_cols = [
    'policy_number','policy_state','insured_zip','incident_location',
    'incident_state','incident_city','insured_hobbies','auto_make',
    'auto_model','auto_year','policy_bind_date','incident_date'
]
data.drop(drop_cols, axis=1, inplace=True)


# ==========================================
# 📌 6. FEATURE ENGINEERING
# ==========================================
data['claim_deviation'] = data['vehicle_claim'] - data['vehicle_claim'].mean()
data['claim_ratio'] = data['vehicle_claim'] / (data['property_claim'] + 1)


# ==========================================
# 📌 7. DEFINE FEATURES & TARGET
# ==========================================
X = data.drop('fraud_reported', axis=1)
y = data['fraud_reported']


# ==========================================
# 📌 8. ENCODING
# ==========================================
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_encoded = pd.get_dummies(X, drop_first=True)


# ==========================================
# 📌 9. TRAIN TEST SPLIT
# ==========================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)


# ==========================================
# 📌 10. MODEL TRAINING
# ==========================================
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=140,
    max_depth=10,
    max_features='sqrt',
    min_samples_split=3,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# 📌 11. MODEL EVALUATION (TEST DATA)
# ==========================================
from sklearn.metrics import accuracy_score, classification_report

y_test_prob = model.predict_proba(X_test)[:,1]
y_test_pred = (y_test_prob > 0.2).astype(int)

print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))


# ==========================================
# 🔥 12. FULL DATASET SCORING (IMPORTANT)
# ==========================================
full_prob = model.predict_proba(X_encoded)[:,1]
full_pred = (full_prob > 0.2).astype(int)


# ==========================================
# 📌 13. CREATE FINAL DATASET
# ==========================================
final_df = data.copy()

final_df['Actual'] = y_encoded
final_df['Fraud_Probability'] = full_prob
final_df['Predicted'] = full_pred


# Risk Segmentation
def risk_level(x):
    if x > 0.7:
        return "High Risk"
    elif x > 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

final_df['Risk_Level'] = final_df['Fraud_Probability'].apply(risk_level)


# ==========================================
# 📌 CONFUSION MATRIX (VISUAL)
# ==========================================
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Labels (important for readability)
labels = ['Not Fraud', 'Fraud']

# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Fraud Detection')
plt.show()

cm_percent = cm / cm.sum() * 100

plt.figure(figsize=(6,5))
sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)

plt.title('Confusion Matrix (%)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==========================================
# 📌 14. EXPORT FOR TABLEAU
# ==========================================
final_df.to_csv("fraud_dashboard_final.csv", index=False)

print("✅ Final dataset ready for Tableau!")