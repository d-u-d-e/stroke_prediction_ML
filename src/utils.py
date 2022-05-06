import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as mtr


def error_rate(y_true, y_pred):
    size = y_true.size
    miss_samples = 0
    for i in range(size):
        if y_pred[i] != y_true[i]:
            miss_samples = miss_samples + 1
    return miss_samples / size


def produce_dataset(url):
    """
    :param url: url of the file processed_dataset.csv
    :return: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(url, header=0)
    X = df[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
            'avg_glucose_level', 'bmi', 'smoking_status']]
    y = df['stroke']

    sm = SMOTE()
    X, y = sm.fit_resample(X, y)

    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), train_size=0.7, random_state=51)

    # Computing the mean and the standard deviation of the training set
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std, y_train, y_test


def compute_scores(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    accuracy_train = mtr.accuracy_score(y_train, y_train_pred)
    accuracy_test = mtr.accuracy_score(y_test, y_test_pred)
    precision_train = mtr.precision_score(y_train, y_train_pred)
    precision_test = mtr.precision_score(y_test, y_test_pred)
    recall_train = mtr.recall_score(y_train, y_train_pred)
    recall_test = mtr.recall_score(y_test, y_test_pred)

    print("SVM accuracy score on training: %f" % accuracy_train)
    print("SVM accuracy score on test: %f" % accuracy_test)
    print("SVM precision score on training: %f" % precision_train)
    print("SVM precision score on test: %f" % precision_test)
    print("SVM recall score on train: %f" % recall_train)
    print("SVM recall score on test: %f" % recall_test)

    return [accuracy_train, accuracy_test, precision_train, precision_test, recall_train, recall_test]
