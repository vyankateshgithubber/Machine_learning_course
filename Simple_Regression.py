# import libraries
import pandas as pd 
# import dataset and assign inpedendent and dependent varible
dataset = pd.read_csv('01Students.csv')

df = dataset.copy()

X = df.iloc[:,:-1]
y = df.iloc[:,-1]
# creatiing the test and train data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
# create regressor 
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
# Train the regressor object on training data
regressor.fit(X_train, y_train)

# predict the test data
y_predict = regressor.predict(X_test)

# R-squared of Regression

slr_score = regressor.score(X_test, y_test)
#print(y_predict,y_test)
print("R squared value", slr_score)
print("coefficient ", regressor.coef_)
print("Intercept ", regressor.intercept_)

# how much error we have - RMSE
from sklearn.metrics import mean_squared_error
import math

rmse = math.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE is ",rmse)


# plot the trendline
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test)
plt.plot(X_test,y_predict)
#plt.ylim(ymin=0)
plt.show()
