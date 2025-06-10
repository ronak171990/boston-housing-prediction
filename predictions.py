import numpy as np
from sklearn.linear_model import LinearRegression

# Load training data (assuming last column is y)
data = np.genfromtxt('x_y_train.csv', delimiter=',')
X_train = data[:, :-1]  # All columns except last
y_train = data[:, -1]   # Last column

# Load test data
X_test = np.genfromtxt('x_test.csv', delimiter=',')

# If X_test is 1D, reshape to 2D
if X_test.ndim == 1:
    X_test = X_test.reshape(-1, X_train.shape[1])

# Train Linear Regression
Ronak = LinearRegression()
Ronak.fit(X_train, y_train)

# Predict on test data
y_pred = Ronak.predict(X_test)

# Save predictions to CSV (5 decimal places, one column, no header)
np.savetxt('predictions.csv', y_pred, fmt='%.5f')