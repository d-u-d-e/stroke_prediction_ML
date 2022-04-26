import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


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
    X = df[['gender', 'age', 'hypertension', 'heart_disease', 'work_type', 'avg_glucose_level', 'bmi']]
    y = df['stroke']

    sm = SMOTE()
    X, y = sm.fit_resample(X, y)

    return train_test_split(X.to_numpy(), y.to_numpy(), train_size=0.7, random_state=51)
