from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
x = housing.data
y = housing.target

x.shape

import pandas as pd
df = pd.DataFrame(x)
print(housing.feature_names)
df.columns = housing.feature_names
df["age_age"] = df.HouseAge **2
df.describe()
X2 = df.values
X2.shape

from sklearn import model_selection
X_train , X_test, Y_train, Y_test = model_selection.train_test_split(x, y, random_state=0)
X2_train , X2_test, Y2_train, Y2_test = model_selection.train_test_split(X2, y, random_state=0)

from sklearn.linear_model import LinearRegression
agl1 = LinearRegression()
agl2 = LinearRegression()

agl1.fit(X_train, Y_train)
agl2.fit(X2_train, Y2_train)


Y_pred = agl1.predict(X_test)
train_score = agl1.score(X_train, Y_train)
test_score = agl1.score(X_test, Y_test)
print("Training Score:", train_score)
print("Testing Score:", test_score)

train2_score = agl2.score(X2_train, Y2_train)
test2_score = agl2.score(X2_test, Y2_test)
print("Training2 Score:", train2_score)
print("Testing2 Score:", test2_score)