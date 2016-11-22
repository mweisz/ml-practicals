from naive_bayes import NBC
from sklearn.linear_model import LogisticRegression

import cPickle as cp
import numpy as np
X, y = cp.load(open(voting.cPikcle,  'rb'))


N, D = X.shape
N_train = int(0.8 * N)
shuffler = np.random.permutation(N)
X_train = X[shuffler[:N_train]]
y_train = y[shuffler[:N_train]]
X_test = X[shuffler[N_train:]]
y_test = y[shuffler[N_train:]]

# Naive Bayes Classifier
# nbc = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)
# nbc.fit(X_train, y_train)

# yhat_nb = nbc.predict(X_test)
# nb_accuracy = np.mean(yhat_nb == y_test)


# Logistic Regression
lgc = LogisticRegression()
lgc.fit(X_train, y_train)
yhat_lg = nbc.predict(X_test)
lg_accuracy = np.mean(yhat_lg == y_test)

print "Naive Bayes vs. Logistic Regression running on Iris dataset:"
# print "\tNB Accuracy: {0}.".format(nb_accuracy)
print "\tLG Accuracy: {0}.".format(lg_accuracy)