import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
from src.utils import produce_dataset
import sklearn.metrics as metrics
from matplotlib import rcParams


def risk_by_age():
    fig = plt.figure(figsize=(10, 5), dpi=150)
    gs = fig.add_gridspec(2, 1)
    gs.update(wspace=0.1, hspace=0.5)
    ax0 = fig.add_subplot(gs[0, 0])
    df = pd.read_csv('../data/processed_dataset.csv', header=0)

    df['age'] = df['age'].astype(int)

    rate = []
    for i in range(df['age'].min(), df['age'].max()):
        rate.append(df[df['age'] < i]['stroke'].sum() / len(df[df['age'] < i]['stroke']))

    sns.lineplot(data=rate, color='#0F4C81', ax=ax0)

    for s in ["top", "right"]:
        ax0.spines[s].set_visible(False)

    plt.title('Risk Increase by Age', fontfamily='serif', fontsize=18, fontweight='bold')

    plt.savefig('../data/risk_by_age.png')


def roc_curve():
    X_train, X_test, y_train, y_test = produce_dataset('../data/processed_dataset.csv')
    lr_model = load('saved_models/log_reg.joblib')
    nn_model = load('saved_models/nn.joblib')
    rf_model = load('saved_models/random_forest.joblib')
    svm_model = load('saved_models/svm.joblib')

    # Plotting the ROC curve
    plt.figure(figsize=(10, 5), dpi=150)
    plt.title('Receiver Operating Characteristic', fontfamily='serif', fontsize=18, fontweight='bold')
    plt.plot([0, 1], [0, 1], 'k--')  # plots the diagonal
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # Computing fpr and tpr for all thresholds of the classification
    # Logistic regression
    probs = lr_model.predict_proba(X_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='LR AUC = %0.2f' % roc_auc)

    # Neural network
    probs = nn_model.predict_proba(X_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'g', label='NN AUC = %0.2f' % roc_auc)

    # Random forest
    probs = rf_model.predict_proba(X_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'r', label='RF AUC = %0.2f' % roc_auc)

    # SVM
    probs = svm_model.predict_proba(X_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'y', label='SVM AUC = %0.2f' % roc_auc)

    plt.legend(loc='lower right')
    plt.savefig('../data/roc_curve.png')


def heatmap():
    dataset = pd.read_csv('../data/processed_dataset.csv')
    plt.figure(figsize=(15, 13), dpi=150)
    plt.title('Heatmap', fontfamily='serif', fontsize=30, fontweight='bold')
    sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm')
    plt.savefig('../data/heatmap.png')


def main():
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    rcParams['axes.titlepad'] = 20
    roc_curve()
    heatmap()
    risk_by_age()


if __name__ == "__main__":
    main()
