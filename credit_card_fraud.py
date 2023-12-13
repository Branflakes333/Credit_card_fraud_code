# Imports
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import numpy as np


def compare_hist(df, param, types):
    type_t = df[df[types] == 1]
    type_f = df[df[types] == 0]

    fig, axes = plt.subplots(1, 3)
    ax = axes.ravel()
    ax[0].hist(df[param], bins=100)
    ax[0].set_title(param, "Total")
    ax[1].hist(type_t[param], bins=100)
    ax[1].set_title(param, types, "=TRUE")
    ax[2].hist(type_f[param], bins=100)
    ax[2].set_title(param, types, "=FALSE")

    plt.tight_layout()
