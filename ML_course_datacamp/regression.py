from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

boston = load_boston()
descr = boston.DESCR

# df = pd.DataFrame(data=boston.data)
csv_loc = 'C:\\Users\\Declan\\Anaconda3\\lib\\site-packages\\sklearn\\datasets\\data\\boston_house_prices.csv'
df = pd.read_csv(csv_loc, skiprows=1)


X = df.drop('MEDV', axis=1).values
y = df['MEDV']
# Try first pred with single feature

# reshape to add a dimension of size 1 to x.
X_rooms = df['RM']
X_rooms = X_rooms.values.reshape(-1, 1)
y = y.values.reshape(-1, 1)

##
plt.scatter(X_rooms, y)
plt.ylabel("House value /$1000")
plt.xlabel("No. of rooms")
##

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_rooms, y)

prediction_space = np.linspace(min(X_rooms), max(X_rooms),100).reshape(-1, 1)

plt.scatter(X_rooms, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space), 
         color='black', linewidth=3)