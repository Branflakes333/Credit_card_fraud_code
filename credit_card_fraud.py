# Imports
import pandas
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def compare_hist(df, param, type):
    type_t = df[df[type] == 1]
    type_f = df[df[type] == 0]

    fig, axes = plt.subplots(1, 3)
    ax = axes.ravel()
    ax[0].hist(df[param], bins=100)
    ax[0].set_title(param, "Total")
    ax[1].hist(type_t[param], bins=100)
    ax[1].set_title(param, type, "=TRUE")
    ax[2].hist(type_f[param], bins=100)
    ax[2].set_title(param, type, "=FALSE")

    plt.tight_layout()


def lasso_train(X, y):
    res = Lasso(alpha=.1).fit(X, y)
    return res


def fraud_detection(model, test_V):
    predict = model.predict(test_V)
    for item in predict:  # catch numbers >1 or <0
        if item >= 0.5:
            item = 1
        else:
            item = 0
    return predict


def repeat_models(X, y, rep):
    sum_acc = 0
    for i in range(rep - 1):
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
        model = lasso_train(train_X, train_y)
        predictions = fraud_detection(model, test_X)
        cnt = np.sum(predictions == test_y)
        sum_acc += cnt / len(predictions)
    print("Average Model Accuracy:", sum_acc / rep)
