import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

# 1. Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

print("MSE:", mean_squared_error(y_true, y_pred))

# 2. Root Mean Squared Error (RMSE)
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print("RMSE:", root_mean_squared_error(y_true, y_pred))

# 3. Cosine Similarity
def compute_cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

vec1 = [1, 2, 3]
vec2 = [4, 5, 6]
print("Cosine Similarity:", compute_cosine_similarity(vec1, vec2))

# 4. Linear Regression
X = np.array([[1], [2], [3], [4], [5]])  # Features
y = np.array([2, 4, 6, 8, 10])  # Labels

model = LinearRegression()
model.fit(X, y)
y_pred_lr = model.predict(X)

print("Linear Regression Predictions:", y_pred_lr)
print("Linear Regression Coefficient:", model.coef_[0])
print("Linear Regression Intercept:", model.intercept_)

# 5. Softmax Function
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtracting max for numerical stability
    return exp_x / exp_x.sum()

values = [2.0, 1.0, 0.1]
print("Softmax Output:", softmax(values))