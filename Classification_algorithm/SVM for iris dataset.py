# import relevant libraries

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
Y = iris.target

# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1234, stratify=Y)

# train the svc
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

svc = SVC(kernel='rbf',gamma=1.0)

svc.fit(X_train, Y_train)

Y_predict = svc.predict(X_test)

cm_rbf01 = confusion_matrix(Y_test, Y_predict)
print(cm_rbf01)

# rbf kernel 
svc = SVC(kernel='rbf',gamma=10)

svc.fit(X_train, Y_train)

Y_predict = svc.predict(X_test)

cm_rbf02 = confusion_matrix(Y_test, Y_predict)
print(cm_rbf02)



# rbf linear kernel 
svc = SVC(kernel='rbf')

svc.fit(X_train, Y_train)

Y_predict = svc.predict(X_test)

cm_linear = confusion_matrix(Y_test, Y_predict)
print(cm_linear)


# Polynomial kernel
svc = SVC(kernel='poly')
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)
cm_poly = confusion_matrix(Y_test, Y_predict)
print(cm_poly)


# Sigmoid Kernel
svc = SVC(kernel='sigmoid')
svc.fit(X_train, Y_train)
Y_predict = svc.predict(X_test)
cm_sig = confusion_matrix(Y_test, Y_predict)
print(cm_sig)
