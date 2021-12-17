import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from functools import reduce  
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from mlxtend.evaluate import bias_variance_decomp
from sklearn import metrics

def table_sum_stats(data_1, data_2):
    """
    Creates Descriptive statistics.
    """
    variables_1 = data_1[list(data_1.columns)]
    variables_2 = data_2[list(data_2.columns)]

    table = pd.DataFrame()
    table["Mean"] = variables_1.mean()
    table["Std. Dev."] = variables_1.std()
    table["variance_mean"]= variables_2.mean()
    table["min"] = variables_1.min()
    table["max"] = variables_1.max()
    table = table.astype(float).round(3)

    return table