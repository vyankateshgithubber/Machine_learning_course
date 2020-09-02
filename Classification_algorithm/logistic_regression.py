# Import Libraries
import pandas as pd

# Read the data  and create a copy
LoanData = pd.read_csv("01Exercise1.csv")

# Identify the missing values
LoanPrep = LoanData.copy()

# Identify the missing values
print(LoanPrep.isnull().sum(axis=0))

# drop rows with missing data
LoanPrep.dropna(inplace=True)

LoanPrep = LoanPrep.drop(['gender'], axis=1)
# Create dummy varibles for categorical varible

print(LoanPrep.dtypes)
LoanPrep = pd.get_dummies(LoanPrep, drop_first=True)

# Normalize data (Income and Loan Amount) using StandardScalar
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

LoanPrep['income'] = scalar.fit_transform(LoanPrep[['income']])

LoanPrep['loanamt'] = scalar.fit_transform(LoanPrep[['loanamt']])

# create X and y
Y = LoanPrep[['status_Y']]
X = LoanPrep.drop(['status_Y'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=1234,stratify=Y)

# Create the logistic regressor
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train, y_train)

y_predict = lr.predict(X_test)

#print(y_test,y_predict)


# Bulid confusion matrix and get the accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)
print(lr.score(X_test, y_test)) 
