from naive_bayes import NBC
import numpy as np

X = np.array([[1, 50],
              [0, 170],
              [0, 250],
              [1, 150],
              [1, 85],
              [0, 60]])

Y = np.array([10, 88, 88, 88, 10, 10])

X_new = np.array([1, 100])

clf = NBC(feature_types=['b', 'r'], num_classes=2)
clf.fit(X, Y)
clf.predict(X_new)
