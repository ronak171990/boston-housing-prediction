import numpy as np
data = np.loadtxt('data.csv', delimiter=',')
x = data[:, 0].reshape(-1, 1)
y = data[:, 1]
from sklearn import model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)
from sklearn.linear_model import LinearRegression
aglr = LinearRegression()
aglr.fit(x_train, y_train)
aglr.coef_
aglr.intercept_
aglr.score(x_test, y_test)
import matplotlib.pyplot as plt
m = aglr.coef_[0]
c = aglr.intercept_
x_line = np.arange(30, 70, 0.1)
y_line = m * x_line + c
plt.plot(x_line, y_line, color='red')
train_id = x_train.reshape(75)
# plt.scatter(x_train, y_train)
# plt.show()

import matplotlib.pyplot as plt
m = aglr.coef_[0]
c = aglr.intercept_
x_line = np.arange(30, 70, 0.1)
y_line = m * x_line + c
plt.plot(x_line, y_line, color='red')
train_id = x_test.reshape(25)
# plt.scatter(x_test, y_test)
# plt.show()

score_test = aglr.score(x_test, y_test)
score_training = aglr.score(x_train, y_train)
print(score_test, score_training)




