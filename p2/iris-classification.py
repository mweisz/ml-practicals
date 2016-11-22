from naive_bayes import NBC
import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris['data'], iris['target']


N, D = X.shape
N_train = int(0.8 * N)
shuffler = np.random.permutation(N)
X_train = X[shuffler[:N_train]]
y_train = y[shuffler[:N_train]]
X_test = X[shuffler[N_train:]]
y_test = y[shuffler[N_train:]]


# Naive Bayes Classifier
nbc = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)
nbc.fit(X_train, y_train)

yhat_nb = nbc.predict(X_test)
nb_accuracy = np.mean(yhat_nb == y_test)


# Logistic Regression
lgc = LogisticRegression()
lgc.fit(X_train, y_train)
yhat_lg = nbc.predict(X_test)
lg_accuracy = np.mean(yhat_lg == y_test)

print "Naive Bayes vs. Logistic Regression running on Iris dataset:"
print "\tNB Accuracy: {0}.".format(nb_accuracy)
print "\tLG Accuracy: {0}.".format(lg_accuracy)

# Learning Curve
for i in range(10):
    shuffler = np.random.permutation(N)
    X_train = X[shuffler[:N_train]]
    y_train = y[shuffler[:N_train]]
    X_test = X[shuffler[N_train:]]
    y_test = y[shuffler[N_train:]]

    accuracies = np.zeros(10)
    for k in range(1, 11):
        train_size = int(1.0*k * N_train)

        nbc.fit(X_train[:train_size], y_train[:train_size])
        yhat_nb = nbc.predict(X_test)
        accuracies[k - 1] = np.mean(yhat_nb == y_test)

    accuracies = (1.0 / 10) * accuracies

print(accuracies)