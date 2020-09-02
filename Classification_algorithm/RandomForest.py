#  Import libraries
import pandas as pd

# Read the dataset
data = pd.read_csv('04 - decisiontreeAdultIncome.csv')

# check for  null values
print(data.isnull().sum(axis=0))


# convert categorcial into dummies
data_prep = pd.get_dummies(data, drop_first=True)

# create X and y
X = data_prep.iloc[:,:-1]
y = data_prep.iloc[:,-1]

# Split the dataset \
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# Import the decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=1234)
dtc.fit(X_train, y_train)
y_predict = dtc.predict(X_test)


# Evaluate the model 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)
score = dtc.score(X_test, y_test)
print(score)


# Import the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1234)
rfc.fit(X_train, y_train)
y_predict = rfc.predict(X_test)


# Evaluate the model 
cm2 = confusion_matrix(y_test, y_predict)
print(cm2)
score2 = dtc.score(X_test, y_test)
print(score2)
