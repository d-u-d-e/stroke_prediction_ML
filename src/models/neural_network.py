import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE

df = pd.read_csv('../../data/processed_dataset.csv', header=0)
X = df[['gender', 'age', 'hypertension', 'heart_disease', 'work_type', 'avg_glucose_level', 'bmi']]
y = df['stroke']

sm = SMOTE()
X, y = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), train_size=0.7, random_state=51)

hl_parameters = {'hidden_layer_sizes': [(50,), (100,), (50, 50,), (100, 100,)]}

mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=300, alpha=1e-4, solver='sgd', tol=1e-4,
                    learning_rate_init=.1)

mlp_cv = GridSearchCV(mlp, hl_parameters)

mlp_cv.fit(X_train, y_train)

print('RESULTS FOR NN\n')

print("Best parameters set found: ", mlp_cv.best_params_)

print("Score with best parameters: ", mlp_cv.best_score_)

print("\nAll scores on the grid: ", mlp_cv.cv_results_['mean_test_score'])

best_mlp = MLPClassifier(hidden_layer_sizes= mlp_cv.best_params_['hidden_layer_sizes'], max_iter=300, alpha=1e-4,
                         solver='sgd', tol=1e-4, learning_rate_init=.1)

best_mlp.fit(X_train, y_train)

training_error = 1 - best_mlp.score(X_train, y_train)

test_error = 1 - best_mlp.score(X_test, y_test)

print("NN training error: %f" % training_error)
print("NN test error: %f" % test_error)
