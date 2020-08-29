# Build Multiple linear regression model

# import libraries
import pandas as pd

# Read data from the csv file
dataset = pd.read_csv('02Students.csv')
df = dataset.copy()

# Split by column for X(independent) and y(dependent) varibles
X = df.iloc[:,:-1]
y = df.iloc[:, -1]

# split by rows for training and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.3,random_state=123)

# Create regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Train the regressor object on training data 
regressor.fit(X_train, y_train)

# predict the test data
y_predict = regressor.predict(X_test)

# R-squared of the regression
mlr_score = regressor.score(X_test, y_test)

# Coefficient and intercept
mlr_coefficient = regressor.coef_
mlr_intercept = regressor.intercept_

print("coefficient  is ",mlr_coefficient)
print("intercept is ",mlr_intercept)
print("Score is ", mlr_score)

from sklearn.metrics import mean_squared_error
import math

mlr_rmse = math.sqrt(mean_squared_error(y_test,y_predict))
print(mlr_rmse)
