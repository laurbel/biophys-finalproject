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








class helperset:
    def __init__(self, data):
        self.data = data
        self.min = np.min(self.data)
        self.max = np.max(self.data)
        self.count = np.count_nonzero(self.data)
        self.mean = np.mean(self.data)


    def get_wavg(self, groupdf):
        self.group =  self.data.groupby(level=0)
        self.avg = np.average(groupdf["building"], weights=groupdf["rating"])
        return self.avg

    
    def get_stats(self, groupdf):
        
        se = pd.DataFrame({"min": groupdf.min(), "max": groupdf.max(),
        "count": groupdf.count(), "mean": groupdf.mean()})
        return se
    
      
    def top(self, n=5, column="rating"):
        return self.data.sort_values(column, ascending=False)[:n]
            
      
