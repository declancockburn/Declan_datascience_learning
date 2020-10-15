import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

# Import clean data
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/module_5_auto.csv'
df = pd.read_csv(path)
df.to_csv('module_5_auto.csv')
df_n = df._get_numeric_data()


def distribution_plot(red_function, blue_function, red_name, blue_name, title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(red_function, hist=False, color='r', label=red_name)
    ax2 = sns.distplot(blue_function, hist=False, color='b', label=blue_name)

    plt.title(title)
    plt.xlabel("Prince ($)")
    plt.ylabel("Proportion of cars")
    plt.show()
    plt.close()


def polly_plot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    xmax = max([xtrain.values.max(), xtest.values.max()])
    xmin = min([xtrain.values.min(), xtest.values.min()])

    x = np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, y_train, 'ro', label="Training Data")
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()

##

y_data = df['price']
x_data = df.drop('price', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=1)

print("number of test samples :", x_test.shape[0])
print("number of training samples:", x_train.shape[0])

# Question 1
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_data, y_data, test_size=0.4, random_state=0)

lre = LinearRegression()
# Fit the model to the training data with just horsepower
lre.fit(x_train[['horsepower']], y_train)
# Then calculate R^2 score on the TEST data
lre.score(x_test[['horsepower']], y_test)
# Compare with the R^2 for the training data.
lre.score(x_train[['horsepower']], y_train)


## Cross validation SCORE

Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
# R^2 across the 4 folds:
Rcross
Rcross.mean()
Rcross.std()

# Get the MSE, by getting the negative mean squared error and multiplying by -1:
-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')

# Cross validation predict

yhat = cross_val_predict(lre, x_data[['horsepower']], y_data, cv=4)
yhat[0:5]