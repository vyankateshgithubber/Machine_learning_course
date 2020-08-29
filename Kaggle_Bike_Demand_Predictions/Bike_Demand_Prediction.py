# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data
bikes = pd.read_csv('hour.csv')

# Analysis and feature selection
bikes_prep = bikes.copy()

bikes_prep = bikes_prep.drop(['index','date','casual','registered'],axis=1)

# Check the null or missing
print(bikes_prep.isnull().sum())

# Simple Visualization of data using Pandas Histogram
bikes_prep.hist(rwidth = 0.9)
plt.tight_layout()
plt.show()


# Visualise the continuous features Vs demand

plt.subplot(2,2,1)
plt.title('Temperature Vs Demand')
plt.scatter(bikes_prep['temp'],bikes_prep['demand'],s=2,c='g')

plt.subplot(2,2,2)
plt.title('aTemp Vs Demand')
plt.scatter(bikes_prep['atemp'],bikes_prep['demand'],s=2,c='b')

plt.subplot(2,2,3)
plt.title('Humidity Vs Demand')
plt.scatter(bikes_prep['humidity'],bikes_prep['demand'],s=2,c='m')

plt.subplot(2,2,4)
plt.title('Windspeed Vs Demand')
plt.scatter(bikes_prep['windspeed'],bikes_prep['demand'],s=2,c='c')
plt.tight_layout()
plt.show()


# Visualize the categorical varible
cat_name = ['season','year','month','hour','holiday','weekday','workingday','weather']
colors = ['m','b','w','r']
for i,data in enumerate(cat_name):
    plt.subplot(3,3,(i+1))
    plt.title('Average Demand Per {}'.format(cat_name[i]))
    cat_list = bikes_prep[cat_name[i]].unique()
    cat_average = bikes_prep.groupby(cat_name[i]).mean()['demand']
    plt.bar(cat_list,cat_average,color=colors)

plt.tight_layout()
plt.show()


# check for outliers
print(bikes_prep['demand'].describe())
print(bikes_prep['demand'].quantile([0.05,0.1,0.15,0.9,0.95,0.99]))

# linearity using correlation coefficient using cor
correlation = bikes_prep[['temp','atemp','humidity','windspeed','demand']].corr()
print(correlation)

# drop
bikes_prep = bikes_prep.drop(['weekday','atemp','windspeed','year','workingday'],axis=1)

# check autocorrelation in demand using acorr plot
df1 = pd.to_numeric(bikes_prep['demand'],downcast='float')
plt.acorr(df1,maxlags=12)
plt.show()

# Log normalise the feature 'Demand'
df1 = bikes_prep['demand']
df2 = np.log(df1)
plt.figure()
df1.hist(rwidth=0.9,bins=20)

plt.figure()
df2.hist(rwidth=0.9,bins=20)
plt.show()

bikes_prep['demand']= np.log(bikes_prep['demand'])

# Autocorrelation in the demand column
t_1 = bikes_prep['demand'].shift(+1).to_frame()
t_1.columns = ['t-1']

t_2 = bikes_prep['demand'].shift(+2).to_frame()
t_2.columns = ['t-2']

t_3 = bikes_prep['demand'].shift(+3).to_frame()
t_3.columns = ['t-3']

bikes_prep_lag = pd.concat([bikes_prep,t_1,t_2,t_3],axis=1)
bikes_prep_lag.dropna(inplace=True)

# Creat Dummy varibles and drop first to avoid dummy
# convert to category type
l = ['season','holiday','weather','month','hour']
for i in l:
    bikes_prep_lag[i] = bikes_prep_lag[i].astype('category')

dummy_df = pd.get_dummies(bikes_prep_lag,drop_first=True)

#  train and test data
# data is time dependent

X = bikes_prep_lag.drop(['demand'],axis=1)
y = bikes_prep_lag[['demand']]

# create training set 70% and test 30%
tr_size = int(0.7 *len(X))

X_train = X.values[0:tr_size]
X_test = X.values[tr_size:len(X)]

y_train = y.values[0:tr_size]
y_test = y.values[tr_size:len(y)]

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

# calculate RMLSE
y_test_e = []
y_predict_e = []
for i in range(0,len(y_test)):
    y_test_e.append(math.exp(y_test[i]))
    y_predict_e.append(math.exp(y_predict[i]))
log_sq_sum = 0.0
for i in range(0,len(y_test_e)):
    log_a = math.log(y_test_e[i] +1)
    log_p = math.log(y_predict_e[i] +1)
    log_diff = (log_p -log_a)**2
    log_sq_sum = log_sq_sum +log_diff
rmsle = math.sqrt(log_sq_sum/len(y_test))
print(rmsle)



