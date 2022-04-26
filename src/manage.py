from joblib import load

from src.utils import produce_dataset


def main():
    X_train, X_test, y_train, y_test = produce_dataset('../data/processed_dataset.csv')

    # ---------------- LR -----------------
    lr_model = load('saved_models/log_reg.joblib')
    # get the training and test error
    training_error = 1. - lr_model.score(X_train, y_train)
    test_error = 1. - lr_model.score(X_test, y_test)

    print("Log reg training error: %f" % training_error)
    print("Log reg test error: %f" % test_error)

    # ---------------- NN -----------------
    nn_model = load('saved_models/nn.joblib')
    # get the training and test error
    training_error = 1. - nn_model.score(X_train, y_train)
    test_error = 1. - nn_model.score(X_test, y_test)

    print("NN training error: %f" % training_error)
    print("NN test error: %f" % test_error)

    # ---------------- RF -----------------
    rf_model = load('saved_models/random_forest.joblib')
    # get the training and test error
    training_error = 1. - rf_model.score(X_train, y_train)
    test_error = 1. - rf_model.score(X_test, y_test)

    print("Random forest training error: %f" % training_error)
    print("Random forest test error: %f" % test_error)

    # ---------------- SVM -----------------
    svm_model = load('saved_models/svm.joblib')

    training_error = 1. - svm_model.score(X_train, y_train)
    test_error = 1. - svm_model.score(X_test, y_test)

    print("SVM training error: %f" % training_error)
    print("SVM test error: %f" % test_error)


if __name__ == "__main__":
    main()
