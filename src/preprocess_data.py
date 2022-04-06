import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../healthcare-dataset-stroke-data.csv', header=0, usecols=range(1, 12))

# Some infos..
# print(df.head(10))
# df.info()
# print(round(df.describe(), 2))
# Filled null values in the column "bmi" with the average
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

df['gender'] = df['gender'].replace({'Male': 0, 'Female': 1, 'Other': -1}).astype(np.uint8)
df['ever_married'] = df['ever_married'].replace({'Yes': 0, 'No': 1, 'Other': -1}).astype(np.uint8)
df['Residence_type'] = df['Residence_type'].replace({'Rural': 0, 'Urban': 1}).astype(np.uint8)
df['work_type'] = df['work_type'].replace(
    {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': -1, 'Never_worked': -2}).astype(np.uint8)
df['smoking_status'] = df['smoking_status'].replace(
    {'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3}).astype(np.uint8)

df.info()
df.to_csv('../data/processed_dataset.csv')

# Creating training and test set
X = df[['gender', 'age', 'hypertension', 'heart_disease', 'work_type', 'avg_glucose_level', 'bmi']]
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=50)

X_train.to_csv('../data/train.csv')
X_test.to_csv('../data/test.csv')

print(df)
