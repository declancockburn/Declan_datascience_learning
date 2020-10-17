import pandas as pd
import numpy as np
import seaborn as sns
from compress_pickle import dump, load

# Uncomment for original data loading:
# df = pd.read_csv('vehicles.csv')
# df = df.drop(['url', 'region', 'vin', 'image_url', 'county', 'lat', 'long', 'id', 'description'], axis=1)
# df.to_pickle('vehicles.pkl')
# dump(df, 'vehicles.gz')

##
df = load('vehicles.gz')

##
# Grab region from URL
df['region'] = df.region_url.str.split("/|\.", n=3, expand=True).iloc[:,2]
df = df.drop('region_url', axis=1)
# Check for null values
missing_vals_count = pd.DataFrame()
missing_vals = pd.DataFrame({'Null': df.isnull().sum().sort_values(ascending=False)})
missing_vals['Percentage'] = (missing_vals['Null']/len(df))*100
missing_vals


##
import re
text = "123.abc,sdf"
re.split(",|\.", text)

##


import _pickle as cpkl
cpkl.dump(df, open('vehicles.cpkl', 'wb'))

##
df.to_pickle('vehicles.pkl')

##


