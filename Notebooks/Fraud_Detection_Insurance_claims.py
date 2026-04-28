# importing required Libraries 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import plotly.express as px
import seaborn as sns 
import plotly

import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')


# reading datasets. 

data= pd.read_csv(r"C:\Users\Hp\Videos\Python,ML & DL\Insuarnce Claims\insurance_claims.csv")

print(data)

# Data Primary information  & basic data cleaning 
data.head()
data.info()

'''Replacing the missing values which are represented by "?" , considering them as null. (applying 
changes directly to the original datasets)'''

data.replace('?', np.nan, inplace = True)

data.describe()

# DATA Pre-Processing 

# checking null values 

data.isna().sum()

# visulaising the null values 

import missingno as msno

msno.bar(data,fontsize=20,color='blue')
plt.show()

# indentifying missing_value column  

missing_columns = data.columns[data.isna().any()]
msno.bar(data[missing_columns],fontsize=20, color = 'orange')

print(missing_columns) #Targeted missing columns 
#Calcualting Missing %

Missing_data = data.isna().sum()/len(data)*100

print(Missing_data)
# Handling Missing Values 
data['collision_type']= data['collision_type'].fillna('unknown',inplace = False)
data['authorities_contacted']=data['authorities_contacted'].fillna('unknown')
data['property_damage']=data['property_damage'].fillna('unknown')
data['police_report_available'] = data['police_report_available'].fillna('unknown')

'''Counting the NaN values after setting them to 0.
The output received is 0 for all columns which means that number of NaN values are 0.'''

data.isna().sum()

#heatmap to visualize the values

plt.figure(figsize = (30,20))
numeric_df = data.select_dtypes(include = ['number'])
corr = numeric_df.corr()  #correlation
sns.set(font_scale = 1.4)
sns.heatmap(data = corr,annot=True, fmt = '.2g', linewidth = 1,cmap='Blues')
plt.show()

updated_data = data.drop('_c39',axis=1)


# removing duplicates from updated dataset 

updated_data= updated_data.drop_duplicates()

print(updated_data)

#Count number of distinct elements in data set
updated_data.nunique()

# checking existence of policy before incident (for record purpose)

updated_data['policy_bind_date']= pd.to_datetime(updated_data['policy_bind_date'])
updated_data['incident_date']= pd.to_datetime(updated_data['incident_date'])

updated_data['policy_age_days'] = ( updated_data['incident_date']- updated_data['policy_bind_date'] ).dt.days

## Removing non-informative or difficult-to-model features such as
# unique IDs, raw location/date fields, and columns with excessive missing values.


to_drop_col = ['policy_number','policy_state','insured_zip','incident_location',
           'incident_state','incident_city','insured_hobbies','auto_make',
           'auto_model','auto_year', 'policy_bind_date','incident_date']

updated_data.drop(to_drop_col, inplace = True, axis  = 1) 

# Fraud vs non-fraud distribution

# Fraud distribution

fraud_count = updated_data['fraud_reported'].value_counts()
print(fraud_count)

# percentage value
fraud_percentage = updated_data['fraud_reported'].value_counts(normalize=True) * 100

ax= sns.countplot(x='fraud_reported', data=updated_data,stat='percent')

for i in ax.containers:
    ax.bar_label(i)
    
plt.show()

# Claim amount patterns
ca=sns.boxplot(x='fraud_reported', y='total_claim_amount', data=data)

medians = data.groupby('fraud_reported')['total_claim_amount'].median()
for i, median in enumerate(medians):
    ca.text(i, median, f'{median:.0f}', ha='center', va='bottom', color='black', fontsize=10)

plt.show()

# # Displaying the data and checking the density of required data fields

plotnumber = 1

num_cols = updated_data.select_dtypes(include=['number']).columns

plt.figure(figsize=(20,18))

for col in num_cols:
    if plotnumber <= 25:
        plt.subplot(5,4,plotnumber)
        sns.histplot(updated_data[col], kde=True)
        plt.xlabel(col, fontsize=10)
        plotnumber += 1

plt.tight_layout()
plt.show()
    
# Outliers Detection 
 
plt.figure(figsize = (20, 18))
plotnumber = 1

for col in num_cols:
    if plotnumber <= 20:
        ax = plt.subplot(5, 4, plotnumber)
        sns.boxplot(updated_data[col])
        plt.xlabel(col, fontsize = 15)
    
    plotnumber += 1
plt.tight_layout()
plt.show()

# Feature Engineering

updated_data['claim_deviation'] = (
    updated_data['vehicle_claim'] - updated_data['vehicle_claim'].mean()
)

updated_data['claim_ratio'] = (
    updated_data['vehicle_claim'] / (updated_data['property_claim'] + 1)
)

''' Feature Selection & DATA Governance '''

# Encording Categorical Variable 

x= updated_data.drop('fraud_reported', axis =1)
y = updated_data['fraud_reported']

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)


lable_y = LabelEncoder()

Y= lable_y.fit_transform(y)

y_train = lable_y.fit_transform(y_train)
y_test = lable_y.transform(y_test)

# checking for multicollinearity

plt.figure(figsize = (18, 12))
numeric_df = updated_data.select_dtypes(include = ['number'])
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))
sns.set(font_scale = 1.2)

sns.heatmap(data = corr, mask = mask, annot = True, fmt = '.2g', linewidth = 1, cmap='coolwarm')
plt.show()

# dropping columns with high multi-collinearity to improve model stability 

to_drop_high_multicollinearitycol = [ 'total_claim_amount', 'claim_deviation']

updated_data.drop(to_drop_high_multicollinearitycol, inplace = True , axis = 1)

# Approaching Random Forest classification to detect fraud in insurance claims 

from sklearn.ensemble import RandomForestClassifier
rndm_clsf = RandomForestClassifier(
    n_estimators=140,
    max_depth=10,
    max_features='sqrt',
    min_samples_split=3,
    min_samples_leaf=1,
    criterion='entropy',   # optional
    class_weight='balanced',  # 🔥 MUST KEEP
    random_state=42
)
rndm_clsf.fit(X_train, y_train)


# probability 

y_prob = rndm_clsf.predict_proba(X_test)[:,1]

Y_pred_new = (y_prob > 0.2).astype(int)
 

results = X_test.copy()

results['Actual'] = y_test
results['Fraud_Probability'] = y_prob
results['Predicted'] = Y_pred_new
results['Risk_Level'] = results['Fraud_Probability'].apply(
    lambda x: 'High Risk' if x > 0.7 else 'Medium Risk' if x > 0.4 else 'Low Risk'
)
plt.figure(figsize=(8,5))

sns.histplot(
    y_prob,
    bins=30,
    kde=False,
    color='seagreen'
   
)
sns.kdeplot(y_prob, color='red', linewidth=2)

plt.title("Fraud Probability Distribution", fontsize=14)
plt.xlabel("Fraud Probability Score")
plt.ylabel("Frequency")

plt.show()

#checking the accuracy_score, confusion matrix and classification report of the model

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

rand_clf_train_acc = accuracy_score(y_train, rndm_clsf.predict(X_train))
rand_clf_test_acc = accuracy_score(y_test, Y_pred_new)

print(f"Training accuracy of Random Forest is : {rand_clf_train_acc}")
print(f"Test accuracy of Random Forest is : {rand_clf_test_acc}")

cm = confusion_matrix(y_test, Y_pred_new)

#Printing the confusion matrix of the model
print(confusion_matrix(y_test, Y_pred_new))
print(classification_report(y_test, Y_pred_new))

# Classification Results: Fraud vs Non‑Fraud

sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
            xticklabels=['Non-Fraud','Fraud'],
            yticklabels=['Non-Fraud','Fraud'])

plt.ylabel('Actual', fontsize=13)
plt.title('Confusion Matrix', fontsize=17, pad=20)
plt.gca().xaxis.set_label_position('top') 
plt.xlabel('Prediction', fontsize=13)
plt.gca().xaxis.tick_top()
plt.gca().figure.subplots_adjust(bottom=0.2)
plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
plt.show()

# Exceuting the precision of the model 

from sklearn.metrics import precision_score, recall_score, precision_recall_curve, average_precision_score

# single-value metrics
prec_score = precision_score(y_test, Y_pred_new)
rec_score = recall_score(y_test, Y_pred_new)

print(f"Precision : {prec_score:.3f}")
print(f"Recall    : {rec_score:.3f}")

# use probabilities for PR curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

sns.lineplot(x=recalls, y=precisions, color='red', linewidth=2)
plt.title(f'Precision-Recall Curve (AP = {ap:.3f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

print(thresholds)

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, color='blue', linewidth=2)
plt.plot([0,1], [0,1], color='gray', linestyle='--')  # baseline
plt.title(f"ROC Curve (AUC = {auc:.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.show()

# Graphical presentation of our model (Random forest)
from sklearn.tree import  plot_tree

tree = rndm_clsf.estimators_[0]
plt.figure(figsize=(20,10))

plot_tree(
    tree,
    feature_names=X_train.columns,
    class_names=['Non-Fraud','Fraud'],
    filled=True,
    max_depth=3,impurity= True
)

plt.show()

#Exporting the decision tree 
from sklearn.tree import export_graphviz
import graphviz


dot_data = export_graphviz(
    tree,
    out_file=None,
    feature_names=X_train.columns,
    class_names=['Non-Fraud','Fraud'],
    filled=True
)

graph = graphviz.Source(dot_data)
graph

graph.render("fraud_tree", format="png", cleanup=True)

# feature importance 
importance = pd.Series(rndm_clsf.feature_importances_, index=X_train.columns)
importance.sort_values(ascending=False).head(10).plot(kind='barh')

plt.title("Top Fraud Drivers")
plt.show()


''' Exporting the refined data to CSV format'''
original_data = updated_data.copy()

# Create final dataset using original data
final_df = X_test.loc[x.index].copy()

# Add model outputs
final_df['Fraud_Probability'] = y_prob
final_df['Predicted'] = Y_pred_new
final_df['Actual'] = y_test

# Add Risk Level
final_df['Risk_Level'] = final_df['Fraud_Probability'].apply(
    lambda x: 'High Risk' if x > 0.7 else 'Medium Risk' if x > 0.4 else 'Low Risk'
)
final_df.to_csv("fraud_dashboard_data.csv", index=False)

updated_data.to_csv('test', index = False)


import pickle

# Save model
pickle.dump(rndm_clsf, open('fraud_model.pkl', 'wb'))

# save columns
pickle.dump(X_train.columns.tolist(), open('columns.pkl', 'wb'))
