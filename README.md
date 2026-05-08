# 🚨 Engineered Insurance Fraud Detection System

## About Project

This project delivers an end-to-end insurance fraud detection system designed to support both predictive decision-making and operational workflow automation. At its core, the solution uses a balanced Random Forest model to identify potentially fraudulent claims with strong performance on imbalanced data, enabling more reliable fraud risk scoring across incoming cases. The trained model is exposed through a Flask API, making it production-ready for seamless integration with downstream applications and internal services. To streamline claim processing, the project incorporates n8n workflow automation for automated claim intake, routing, and orchestration across the pipeline. For visibility and monitoring, Tableau dashboards provide clear, interactive reporting on model outputs, fraud trends, and operational metrics, helping stakeholders track performance and investigate high-risk claims efficiently. Together, these components create a scalable, modular, and business-ready fraud detection system built for practical deployment in real-world insurance workflows.

---

# 📌 Project Overview

The project covers the complete fraud detection workflow:

* Data cleaning & preprocessing
* Exploratory data analysis (EDA)
* Feature engineering
* Fraud prediction using Machine Learning
* Fraud probability scoring
* Risk classification
* Dashboard-ready output generation
* Flask API deployment for model serving and downstream integration
* n8n workflow automation for claim intake and orchestration
* Tableau dashboards for monitoring and operational reporting

The model predicts the likelihood of fraud and classifies claims into:

* 🔴 High Risk
* 🟠 Medium Risk
* 🟢 Low Risk

---

# 🧠 Business Problem

Insurance fraud leads to major financial losses and increases claim investigation costs.

The goal of this project is to:

* Detect suspicious claims early
* Support faster investigations
* Reduce fraud leakage
* Improve operational efficiency

---

# ⚙️ Technologies Used

* Python
* Pandas & NumPy
* Matplotlib & Seaborn
* Plotly
* Scikit-learn
* Graphviz
* Flask
* n8n
* Tableau

---

# 🔄 Workflow

```text
Insurance Claims Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Data Encoding
        ↓
Random Forest Model
        ↓
Fraud Probability Score
        ↓
Risk Classification
        ↓
Dashboard Output
```

---

# 🛠 Key Features

### ✔ Missing Value Handling

* Replaced invalid values (`?`) with nulls
* Filled missing categorical fields using business logic

### ✔ Feature Engineering

Created custom features such as:

* Claim deviation
* Claim ratio
* Policy age

### ✔ Fraud Probability Scoring

Generated fraud probability instead of only binary predictions.

### ✔ Risk-Based Classification

Claims are categorized into:

* High Risk
* Medium Risk
* Low Risk

### ✔ Dashboard-Ready Output

Final predictions are exported into CSV format for reporting and dashboard creation.

### ✔ Production Components

* Balanced Random Forest model for fraud detection and risk scoring
* Flask API for model serving and downstream application integration
* n8n workflow automation for claim intake, routing, and orchestration
* Tableau dashboards for monitoring claim performance and fraud trends

---

### 📊 Project Dashboards

![Model Dashboard](Dashboards/Model%20Dashboard.jpg)

![Performance Dashboard](Dashboards/Performance%20Dashboard.jpg)

---

# 🤖 Machine Learning Model

The project uses a **balanced Random Forest Classifier** for fraud detection.

Key configurations:

* Balanced class weighting
* Controlled tree depth
* Entropy-based splitting

The model was trained to handle imbalanced fraud data effectively.

---

# 🎯 Threshold Optimization

Instead of using the default threshold (`0.5`),
a custom threshold of:

```python
0.2
```

was used to improve fraud detection recall.

This approach helps capture more potentially fraudulent claims, which is critical in fraud analytics.

---

# 🚦 Risk Segmentation

| Fraud Probability | Risk Level  |
| ----------------- | ----------- |
| > 0.7             | High Risk   |
| 0.4 – 0.7         | Medium Risk |
| < 0.4             | Low Risk    |

---

# 📊 Model Evaluation

The project includes:

* Accuracy Score
* Precision & Recall
* Confusion Matrix
* ROC-AUC Curve
* Precision-Recall Curve
* Feature Importance Analysis

---

# 📈 Visualizations Included

* Fraud Distribution
* Claim Amount Analysis
* Correlation Heatmaps
* Fraud Probability Distribution
* ROC Curve
* Precision-Recall Curve
* Feature Importance Graph

---

# 📦 Output Files

| File                       | Description                     |
| -------------------------- | ------------------------------- |
| `fraud_model.pkl`          | Trained ML model                |
| `columns.pkl`              | Saved feature columns           |
| `fraud_dashboard_data.csv` | Dashboard-ready prediction data |

---

# 🚀 Future Improvements

* Power BI dashboard integration
* Cloud deployment
* Advanced ML models (XGBoost / LightGBM)

---

# ▶️ How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the model

```bash
python scripts/fraud_model.py
```

---

# 👨‍💻 About This Project

This project was developed as part of a portfolio focused on:

* Fraud Detection
* Insurance Analytics
* Machine Learning
* Data Automation
* Business Intelligence
