from sklearn import linear_model
from joblib import dump

from src.utils import produce_dataset

X_train, X_test, y_train, y_test = produce_dataset('../../data/processed_dataset.csv')

# Defining the logistic regression model
log_reg = linear_model.LogisticRegression(C=1e5, max_iter=1000)

# Learning from training set
model = log_reg.fit(X_train, y_train)

# Saving the model
dump(log_reg, '../saved_models/log_reg.joblib')
