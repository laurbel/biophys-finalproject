from unicodedata import category
import matplotlib.pyplot as plt
import os
import sqlite3
import unittest
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()
import numpy as np
from pathlib import Path

#our helper functions
def top(df, n=5, column="rating"):
    return df.sort_values(column, ascending=False)[:n]


def get_stats(group):
    return pd.DataFrame(
        {"min": group.min(), "max": group.max(),
        "count": group.count(), "mean": group.mean()}
    )

def get_wavg(group):
    return np.average(group["building"], weights=group["rating"])

def normalize(x):
    return (x - x.mean()) / x.std()


def AutoRegModel(data, lags=int): 
    MAXLAGS = lags
    model = AutoReg(list(data['rating']), MAXLAGS)
    results = model.fit()
    return results.summary()