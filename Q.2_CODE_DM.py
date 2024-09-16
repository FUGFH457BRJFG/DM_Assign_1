import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


# Load your dataset
data = pd.read_csv('/Users/shristisingh/Philadelphia_Crime_Rate_noNA.csv') # Replace with your dataset file path

# View the first few rows of the dataset to understand its structure
print(data.head())

# Define features (X) and target variable (y)
# Replace 'Price' with the actual name of your target variable
X = data.drop(columns=['HousePrice', 'County', 'Name'])  # Drop 'Price', 'country', and 'name'
y = data['HousePrice']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

# Optionally, print the model's coefficients and intercept
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')




########
# Test for non-linear relationships using polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Training a new model on the polynomial features
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

# Making predictions and evaluating for results
y_pred_poly = model_poly.predict(X_test_poly)
r2_poly = r2_score(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)

print(f'Polynomial R-squared: {r2_poly}')
print(f'Polynomial Mean Squared Error: {mse_poly}')


