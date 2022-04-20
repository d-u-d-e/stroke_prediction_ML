import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE

df = pd.read_csv('../../data/processed_dataset.csv', header=0)
X = df[['gender', 'age', 'hypertension', 'heart_disease', 'work_type', 'avg_glucose_level', 'bmi']]
y = df['stroke']

sm = SMOTE()
X, y = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), train_size=0.7, random_state=51)

clf = RandomForestClassifier(max_depth=2, random_state=0)

rf_parameters = {'max_depth': list(range(1, 11)),
                 'n_estimators': list(range(10, 200, 10)),
                 'bootstrap': [True, False]}

rf_cv = GridSearchCV(clf, rf_parameters)

rf_cv.fit(X_train, y_train)

print('RESULTS FOR RANDOM FOREST\n')
print("Best parameters set found: ", rf_cv.best_params_)
print("Score with best parameters: ", rf_cv.best_score_)
print("\nAll scores on the grid: ", rf_cv.cv_results_['mean_test_score'])

best_clf = RandomForestClassifier(max_depth=rf_cv.best_params_['max_depth'], random_state=0)

best_clf.fit(X_train, y_train)

print("Training accuracy: ", best_clf.score(X_train, y_train))
print("Test accuracy: ", best_clf.score(X_test, y_test))