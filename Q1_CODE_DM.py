import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

#generating synthatic data
np.random.seed(42)
n_samples = 30
X = np.sort(np.random.rand(n_samples))       # Random x values in [0, 1]
y_true = np.log(X)                           # True y values (let the true function be log(x))
noise = np.random.normal(0, 0.3, n_samples)  # Random noise 
y = y_true + noise                           # generating Noisy observations 


# Linear Model with degree=1
model_1 = make_pipeline(PolynomialFeatures(1), LinearRegression())
model_1.fit(X[:, np.newaxis], y)

# Moderate Complexity Model with degree = 3
model_3 = make_pipeline(PolynomialFeatures(3), LinearRegression())
model_3.fit(X[:, np.newaxis], y)

# Complex Model with degree =10
model_10 = make_pipeline(PolynomialFeatures(10), LinearRegression())
model_10.fit(X[:, np.newaxis], y)

# predictiong value of y for different model
X_plot = np.linspace(0, 1, 100)
y_pred_1 = model_1.predict(X_plot[:, np.newaxis])
y_pred_3 = model_3.predict(X_plot[:, np.newaxis])
y_pred_10 = model_10.predict(X_plot[:, np.newaxis])

#plot to show bias variance tradeoff

plt.figure(figsize=(14, 6))
plt.scatter(X, y, color='black', label="Data with Noise")
plt.plot(X_plot, np.log(X_plot), color='green', linewidth=2, label="True Function")
plt.plot(X_plot, y_pred_1, color='red', linewidth=2, label="Linear Model (degree 1)")
plt.plot(X_plot, y_pred_3, color='blue', linewidth=2, label="Moderate Model (degree 3)")
plt.plot(X_plot, y_pred_10, color='purple', linewidth=2, label="Complex Model (degree 10)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Bias-Variance Tradeoff")
plt.legend()
plt.show()
