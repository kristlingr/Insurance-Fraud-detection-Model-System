import sys
import json
import pickle
import pandas as pd

model = pickle.load(open('fraud_model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))


if len(sys.argv) > 1:
    input_data = json.loads(sys.argv[1])
else:
    input_data = {
        "months_as_customer": 328,
        "age": 48,
        "policy_annual_premium": 1406.91
    }

df = pd.DataFrame([input_data])
df = pd.get_dummies(df)
df = df.reindex(columns=columns, fill_value=0)

prob = model.predict_proba(df)[0][1]

if prob > 0.7:
    risk = "HIGH"
elif prob > 0.4:
    risk = "MEDIUM"
else:
    risk = "LOW"

result = {
    "Fraud_Probability": float(prob),
    "Risk_Level": risk
}

print(result)