# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


# reading data from datasheet
data = '/Users/shristisingh/Downloads/wine (1)/wine.data'
columns = ['WineType', 'Alcohol', 'MalicAcid', 'Ash', 'AlcalinityOfAsh', 'Magnesium', 'TotalPhenols', 'Flavanoids', 'NonflavanoidPhenols', 'Proanthocyanins', 'ColorIntensity', 'Hue', 'OD280/OD315', 'Proline']

df = pd.read_csv(data, header=None, names=columns)

# Checking if the dataset was loaded successfully
print(df.head())


#  Data Preprocessing: X and y 
X = df.drop('WineType', axis=1)  # Features
y = df['WineType']               # Target

# Standardizing the data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Spliting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#  Building Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predicting on the test set fir logistic regression
y_pred_logreg = log_reg.predict(X_test)

# Building KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predicting on the test set for KNN
y_pred_knn = knn.predict(X_test)

# printing results from both models
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))

print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

#  Comparing and printing the Best Model
if accuracy_score(y_test, y_pred_logreg) > accuracy_score(y_test, y_pred_knn):
    print("\nLogistic Regression performs better.")
else:
    print("\nKNN performs better.")
