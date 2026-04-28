# Insurance-Fraud-detection-Model-System
# 🚨 Insurance Fraud Detection System

## 📌 Overview
This project builds an end-to-end Machine Learning pipeline to detect fraudulent insurance claims.

The model predicts fraud probability and classifies claims into:
- Low Risk
- Medium Risk
- High Risk

---

## ⚙️ Tech Stack
- Python (Pandas, NumPy, Scikit-learn)
- Data Visualization (Seaborn, Matplotlib, Plotly)
- Machine Learning (Random Forest)

---

## 🔄 Workflow

1. Data Cleaning
   - Replaced "?" with null values
   - Handled missing values using domain logic

2. Feature Engineering
   - Policy age calculation
   - Claim deviation & claim ratio

3. Data Preprocessing
   - One-hot encoding
   - Train-test split (stratified)

4. Model Building
   - Random Forest Classifier
   - Class imbalance handled using `class_weight='balanced'`

5. Prediction Logic
   - Fraud Probability generated
   - Custom threshold applied: **0.2**

6. Risk Classification
   - High Risk → Probability > 0.7
   - Medium Risk → 0.4 – 0.7
   - Low Risk → < 0.4

---

## 📊 Key Insights

- Fraud cases are highly imbalanced
- Lower threshold improves fraud detection (recall)
- Claim amount patterns differ significantly in fraud vs non-fraud

---

## 📈 Model Performance

- Accuracy Score
- Precision & Recall
- ROC-AUC Curve
- Confusion Matrix

---

## 📸 Visualizations

- Fraud Distribution
- Claim Amount Analysis
- Feature Importance
- ROC Curve
- Precision-Recall Curve

---

## 🚀 Outputs

- Fraud Probability Score
- Risk Segmentation
- Dashboard-ready dataset

---

## 📦 Files Generated

- `fraud_model.pkl` → Trained model
- `columns.pkl` → Feature columns
- `fraud_dashboard_data.csv` → Final output dataset

---

## 🔮 Future Improvements

- Real-time fraud detection API
- Integration with n8n workflow
- Power BI dashboard automation
- Model tuning (XGBoost, LightGBM)

---

## 💡 Business Impact

- Early fraud detection
- Reduced claim leakage
- Automated risk scoring

