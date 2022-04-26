from src.utils import produce_dataset
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from joblib import dump

X_train, X_test, y_train, y_test = produce_dataset('../../data/processed_dataset.csv')

# Cross validation
parameters = {'C': [1, 10, 100], 'gamma': [0.01, 0.1, 1.]}

rbf_SVM = SVC(kernel='rbf')

svm_cv = GridSearchCV(rbf_SVM, parameters, verbose=1)

svm_cv.fit(X_train, y_train)

print('\nRESULTS FOR rbf KERNEL\n')
print("Best parameters set found: ", svm_cv.best_params_)
print("Score with best parameters: ", svm_cv.best_score_)
print("\nAll scores on the grid: ", svm_cv.cv_results_['mean_test_score'])

# Defining the model w.r.t. the best parameters found
svm_model = SVC(kernel='rbf', C=svm_cv.best_params_['C'], gamma=svm_cv.best_params_['gamma'], verbose=True)

# Training the model
svm_model.fit(X_train, y_train)

# Saving the model
dump(svm_model, '../saved_models/svm.joblib')
