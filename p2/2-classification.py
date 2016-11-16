from naive_bayes import NBC
import numpy as np

X = np.array([[1, 50],
              [0, 170],
              [1, 150],
              [1, 85]])

Y = np.array([10, 10, 88, 10])

clf = NBC()
clf.fit(X, Y)
# clf.predict(1)
