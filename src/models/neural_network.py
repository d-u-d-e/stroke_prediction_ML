from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from joblib import dump

from src.utils import produce_dataset

X_train, X_test, y_train, y_test = produce_dataset('../../data/processed_dataset.csv')

# Cross validation
hl_parameters = {'hidden_layer_sizes': [(40, 40, 40,), (50, 50, 50), (60, 60, 60)]}

mlp = MLPClassifier(max_iter=500, alpha=1e-4, solver='sgd', tol=1e-4, verbose=True, learning_rate_init=0.01,
                    learning_rate='adaptive')

mlp_cv = GridSearchCV(mlp, hl_parameters, verbose=1)

mlp_cv.fit(X_train, y_train)

print('RESULTS FOR NN\n')
print("Best parameters set found: ", mlp_cv.best_params_)
print("Score with best parameters: ", mlp_cv.best_score_)
print("\nAll scores on the grid: ", mlp_cv.cv_results_['mean_test_score'])

# Defining the model w.r.t. the best parameters found
mlp_model = MLPClassifier(hidden_layer_sizes=mlp_cv.best_params_['hidden_layer_sizes'], max_iter=300, alpha=1e-4,
                          solver='sgd', tol=1e-4, learning_rate_init=.1)

# Training the model
mlp_model.fit(X_train, y_train)

# Saving the model
dump(mlp_model, '../saved_models/nn.joblib')
