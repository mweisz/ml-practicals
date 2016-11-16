from naive_bayes import NBC
import numpy as np

from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris['data'], iris['target']

N, __ = X.shape
N_train = int(N * 0.8)
N_test = N - N_train

X_train = X[:N_train]
y_train = y[:N_train]

X_test = X[N_train:]
y_test = y[N_train:]


print "{} -> {}".format(X_train[0, :], y_train[0])

nbc = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)
nbc.fit(X_train, y_train)

correct = 0
for i, x in enumerate(X_test):
    yhat = nbc.predict(x)
    print "Predicted class {}".format(yhat)
    print "Actual class {}\n\n".format(y_test[i])
    if yhat == y_test[i]:
        correct += 1

accuracy = float(correct) / y_test.size
print(accuracy)

# yhat = nbc.predict(X_test)
# test_accuracy = np.mean(yhat == y_test)
# print(test_accuracy)
# print "Actual class {}".format(y_test[0])