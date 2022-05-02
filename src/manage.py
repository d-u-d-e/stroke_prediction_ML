from joblib import load

from src.utils import produce_dataset, compute_scores
import pandas as pd


def main():
    X_train, X_test, y_train, y_test = produce_dataset('../data/processed_dataset.csv')

    # ---------------- LR -----------------
    lr_model = load('saved_models/log_reg.joblib')
    lr_scores = compute_scores(lr_model, X_train, X_test, y_train, y_test)
    print("----------------")

    # ---------------- NN -----------------
    nn_model = load('saved_models/nn.joblib')
    nn_scores = compute_scores(nn_model, X_train, X_test, y_train, y_test)
    print("----------------")

    # ---------------- RF -----------------
    rf_model = load('saved_models/random_forest.joblib')
    rf_scores = compute_scores(rf_model, X_train, X_test, y_train, y_test)
    print("----------------")

    # ---------------- SVM -----------------
    svm_model = load('saved_models/svm.joblib')
    svm_scores = compute_scores(svm_model, X_train, X_test, y_train, y_test)
    print("----------------")

    # Creating dataframe with scores
    data = {'Accuracy on train': [lr_scores[0], nn_scores[0], rf_scores[0], svm_scores[0]],
            'Accuracy on test': [lr_scores[1], nn_scores[1], rf_scores[1], svm_scores[1]],
            'Precision on train': [lr_scores[2], nn_scores[2], rf_scores[2], svm_scores[2]],
            'Precision on test': [lr_scores[3], nn_scores[3], rf_scores[3], svm_scores[3]],
            'Recall on train': [lr_scores[4], nn_scores[4], rf_scores[4], svm_scores[4]],
            'Recall on test': [lr_scores[5], nn_scores[5], rf_scores[5], svm_scores[5]]}
    df = pd.DataFrame(data,
                      index=['Logistic regression', 'Multilayer perceptron', 'Random forest', 'Support vector machine'])
    df.to_csv('../data/scores.csv')


if __name__ == "__main__":
    main()
