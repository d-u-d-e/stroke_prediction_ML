import matplotlib.pyplot as plt
from joblib import load
from src.utils import produce_dataset
import sklearn.metrics as metrics


X_train, X_test, y_train, y_test = produce_dataset('../data/processed_dataset.csv')
lr_model = load('saved_models/log_reg.joblib')
nn_model = load('saved_models/nn.joblib')
rf_model = load('saved_models/random_forest.joblib')
svm_model = load('saved_models/svm.joblib')

# Plotting the ROC curve
plt.title('Receiver Operating Characteristic')
#plt.plot([0, 1], [0, 1],'r--')  #plots the diagonal
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# Computing fpr and tpr for all thresholds of the classification
# Logistic regression
probs = lr_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label='LR AUC = %0.2f' % roc_auc)

# Neural network
probs = nn_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, 'g', label='NN AUC = %0.2f' % roc_auc)

# Random forest
probs = nn_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, 'r', label='RF AUC = %0.2f' % roc_auc)

# SVM
probs = nn_model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, 'y', label='SVM AUC = %0.2f' % roc_auc)

plt.legend(loc='lower right')
plt.show()


# PLOTTING THE HEATMAP OF OUR DATASET'S CORRELATION MATRIX
#import seaborn as sns
#import pandas as pd
#dataset = pd.read_csv('../data/processed_dataset.csv')
#sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
#plt.show()