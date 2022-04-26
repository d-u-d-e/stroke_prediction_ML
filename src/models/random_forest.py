from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from joblib import dump

from src.utils import produce_dataset

X_train, X_test, y_train, y_test = produce_dataset('../../data/processed_dataset.csv')

# Cross validation
rf_parameters = {'max_depth': [1, 5, 10],
                 'n_estimators': [50, 100, 150, 200],
                 'bootstrap': [True, False]}

clf = RandomForestClassifier(random_state=0)

rf_cv = GridSearchCV(clf, rf_parameters, verbose=1)

rf_cv.fit(X_train, y_train)

print('RESULTS FOR RANDOM FOREST\n')
print("Best parameters set found: ", rf_cv.best_params_)
print("Score with best parameters: ", rf_cv.best_score_)
print("\nAll scores on the grid: ", rf_cv.cv_results_['mean_test_score'])

# Defining the model w.r.t. the best parameters found
rf_model = RandomForestClassifier(max_depth=rf_cv.best_params_['max_depth'],
                                  n_estimators=rf_cv.best_params_['n_estimators'],
                                  bootstrap=rf_cv.best_params_['bootstrap'], random_state=0)

# Training the model
rf_model.fit(X_train, y_train)

# Saving the model
dump(rf_model, '../saved_models/random_forest.joblib')
