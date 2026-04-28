# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:21:55 2026

@author: Hp
"""

from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open('fraud_model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    prob = model.predict_proba(df)[0][1]

    if prob > 0.7:
        risk = "HIGH"
    elif prob > 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return jsonify({
        "Fraud_Probability": float(prob),
        "Risk_Level": risk
    })

if __name__ == "__main__":
    app.run(port=5000)