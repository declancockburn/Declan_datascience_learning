import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv("kc_house_data_NaN.csv")

df.columns

df = df.drop(['Unnamed: 0', 'id'], axis=1)
