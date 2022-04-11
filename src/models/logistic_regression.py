import pandas as pd
from src import utils as utils
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Creating training and test set

df = pd.read_csv('../../data/processed_dataset.csv', header=0)
X = df[['gender', 'age', 'hypertension', 'heart_disease', 'work_type', 'avg_glucose_level', 'bmi']]
y = df['stroke']

#
sm = SMOTE()
X, y = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), train_size=0.7, random_state=51)

# defining the logistic regression model
logreg = linear_model.LogisticRegression(C=1e5, max_iter=1000)

# learn from training set

model = logreg.fit(X_train, y_train)

# predict on training set

pred = model.predict(X_train)

# print the error rate = fraction of misclassified samples
print("Error rate on training set: " + str(utils.error_rate(y_train, pred)))
# accuracy_training = accuracy_score(y_train, pred)
# print("Error rate on training set: " + str(1-accuracy_training))

# predict on test set

pred = model.predict(X_test)

# print the error rate = fraction of misclassified samples
print("Error rate on test set: " + str(utils.error_rate(y_test, pred)))
# accuracy_test = accuracy_score(y_test, pred)
# print("Error rate on training set: " + str(1-accuracy_test))

