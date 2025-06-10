import numpy as np
from sklearn.model_selection import train_test_split

data = np.loadtxt('FuelEconomy.csv', delimiter=',')
X = data[:, 0]
y = data[:, 1]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3, random_state=42)



def fit(x_train, y_train):
    x_mean = np.mean(x_train)
    y_mean = np.mean(y_train)
    m = np.sum((x_train - x_mean) * (y_train - y_mean)) / np.sum((x_train - x_mean)**2)
    c = y_mean - m * x_mean
    return m, c

def predict(x, m, c):
    return new_func(x, m, c)

def new_func(x, m, c):
    return m * x + c

def score(y_truth, y_pred):
    ss_res = np.sum((y_truth - y_pred) ** 2)
    ss_tot = np.sum((y_truth - np.mean(y_truth)) ** 2)
    return 1 - (ss_res / ss_tot)

def cost(x, y, m, c):
    y_pred = predict(x, m, c)
    return np.mean((y - y_pred) ** 2)

m, c = fit(X_train, Y_train)

y_train_pred = predict(X_train, m, c)
y_test_pred = predict(X_test, m, c)


print(round(score(X_train, y_train_pred), 3))  # Training score
print(round(score(Y_test, y_test_pred), 3))    # Testing score
print(round(cost(X_test, Y_test, m, c), 3))    # Cost on test data
