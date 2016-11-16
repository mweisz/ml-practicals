import numpy as np
from scipy.stats import norm
from operator import itemgetter


class NBC:
    def __init__(self, feature_types=None, num_classes=None):
        """NBC(feature_types=['b', 'r', 'b'], num_classes=4)"""
        self.num_classes = num_classes
        self.feature_types = feature_types
        self.is_fit = False

    def fit(self, X, Y):
        self.classes = self.create_classes(X, Y)
        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            raise Exception('Need to fit model before making predictions.')

        # TODO: For now we assume X is just one datapoint not a matrix.
        max_probability = 0
        predicted_class_label = self.classes[0]['label']
        for c in self.classes:
            p = self.compute_class_probability(c, X)
            print "P(Y={0} | {1}) = {2}".format(c['label'], X, p)
            if p > max_probability:
                max_probability = p
                predicted_class_label = c['label']

        return predicted_class_label

    def compute_class_probability(self, c, x):
        """ Pr(Y=c|x) """

        probabilites = []

        D = x.size
        for i in range(D):
            if self.feature_types[i] == 'b':
                bernoulli = c['means'][i] if x[i] == 1 else 1 - c['means'][i]
                probabilites.append(bernoulli)
            elif self.feature_types[i] == 'r':
                gauss = norm.pdf(x[i], c['means'][i], c['stds'][i])
                probabilites.append(gauss)

        return np.prod(probabilites)

    def create_classes(self, X, Y):
        class_labels = np.unique(Y)
        if len(class_labels) != self.num_classes:
            raise Exception('Number of classes does not match data.')

        classes = []

        for class_label in class_labels:
            idx = (Y == class_label)

            means = np.mean(X[idx], axis=0)
            stds = np.std(X[idx], axis=0)

            classes.append({'label': class_label,
                           'means': means, 'stds': stds})
        return classes
