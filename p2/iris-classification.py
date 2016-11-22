import sys
from naive_bayes import NBC
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

# Using seaborn as default style for matplotlib
import seaborn
seaborn.set()

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
yhat_lg = lgc.predict(X_test)
lg_accuracy = np.mean(yhat_lg == y_test)

print "Naive Bayes vs. Logistic Regression running on Iris dataset:"
print "\tNB Accuracy: {0}.".format(nb_accuracy)
print "\tLG Accuracy: {0}.".format(lg_accuracy)

# Learning Curve
MAX_TRAILS = 200
CLASSIFIER_COUNT = 10
accuracies_nb = np.zeros(CLASSIFIER_COUNT)
accuracies_lg = np.zeros(CLASSIFIER_COUNT)

for k in range(1, CLASSIFIER_COUNT + 1):
    N_train_k = int(0.1 * k * N_train)

    for i in range(MAX_TRAILS):
        # Print status
        print "Training: {0}/{1}... \r".format(k, CLASSIFIER_COUNT),
        sys.stdout.flush()

        shuffler = np.random.permutation(N)
        Xtrain = X[shuffler[:N_train_k]]
        ytrain = y[shuffler[:N_train_k]]
        Xtest = X[shuffler[N_train_k:]]
        ytest = y[shuffler[N_train_k:]]

        nbc = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)
        nbc.fit(Xtrain[:N_train_k], ytrain[:N_train_k])
        yhat_nb = nbc.predict(Xtest)
        accuracies_nb[k - 1] += np.mean(yhat_nb == ytest)

        lgc = LogisticRegression()
        lgc.fit(Xtrain[:N_train_k], ytrain[:N_train_k])
        yhat_lg = lgc.predict(Xtest)
        accuracies_lg[k - 1] += np.mean(yhat_lg == ytest)

accuracies_nb = (1.0 / MAX_TRAILS) * accuracies_nb
accuracies_lg = (1.0 / MAX_TRAILS) * accuracies_lg

print(accuracies_nb)
print(accuracies_lg)

plt.title('Iris: Learning Curve')
plt.ylabel('Accuracy')
plt.plot(range(1, CLASSIFIER_COUNT + 1), accuracies_nb, label="Naive Bayes")
plt.plot(range(1, CLASSIFIER_COUNT + 1), accuracies_lg, label="Logistic Regression")
plt.legend(loc='lower right')
plt.show()
