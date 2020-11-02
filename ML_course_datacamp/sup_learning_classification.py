from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

iris = datasets.load_iris()

print(iris.keys())
print(type(iris.data))
print(type(iris.target))


X = iris.data
y = iris.target

df = pd.DataFrame(X,columns=iris.feature_names)

_ = pd.plotting.scatter_matrix(df, c=y, figsize=[7,7], s=100, marker='D')
##
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(iris['data'], iris['target'])

##

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=21, stratify=y)
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(knn.score(X_test, y_test))




