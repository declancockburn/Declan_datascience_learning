import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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
    # ax1 = sns.kdeplot(red_function, hist=False, color='r', label=red_name)
    # ax2 = sns.kdeplot(blue_function, hist=False, color='b', label=blue_name)

    plt.title(title)
    plt.xlabel("Prince ($)")
    plt.ylabel("Proportion of cars")
    plt.show()
    plt.legend()
    # plt.close()


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

##
#Part 2 from wk5 lab
lr = LinearRegression()
x_cols = ['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']
lr.fit(x_train[x_cols], y_train)

lr.score(x_train[x_cols], y_train)
yhat_train = lr.predict(x_train[x_cols])
yhat_train[:5]


lr.score(x_test[x_cols], y_test)
yhat_test = lr.predict(x_test[x_cols])
yhat_test[:5]

Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
distribution_plot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
distribution_plot(y_test, yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)

##
# Overfitting, fitting the noise it was above. Let's make a 5th degree poly transformation of horsepower(??) with 55%
# data training

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)

pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])

# Now lets fit to this:
poly = LinearRegression()
poly.fit(x_train_pr, y_train)
# poly.score(x_train_pr,y_train)
# poly.score(x_test_pr,y_test)

yhat = poly.predict(x_test_pr)
yhat[:5]

print("Predicted values:", yhat[0:4])
print("True values:", y_test[0:4].values)

polly_plot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly, pr)

poly.score(x_train_pr, y_train)
poly.score(x_test_pr, y_test)

## Try different orders of polynomials and see how R2 changes

r_squared_test = []
order = range(1,5)
for n in order:
    pr = PolynomialFeatures(degree=n)
    # Create polynomial features
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    lr.fit(x_train_pr, y_train)
    r_squared_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, r_squared_test)

## Multiple Poly features regression

pr1 = PolynomialFeatures(degree=2)
x_train_pr1 = pr1.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
x_test_pr1 = pr1.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

poly1 = LinearRegression()
poly1.fit(x_train_pr1, y_train)

yhat_train = poly1.predict(x_train_pr1)
distribution_plot(yhat_train, y_train, 'predicted', 'actual', 'Prediction of training data')


yhat_test = poly1.predict(x_test_pr1)
distribution_plot(yhat_test, y_test, 'predicted', 'actual', 'Prediction of test data')











