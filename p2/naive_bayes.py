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

        probs = np.apply_along_axis(self.compute_class_probabilites, 1, X)
        max_class_idx = np.argmax(probs, axis=1)
        return np.array([self.classes[i]['label'] for i in max_class_idx])

    def compute_class_probabilites(self, X):
        class_probabilities = np.zeros(X.size)
        for i, c in enumerate(self.classes):
            class_probabilities[i] = self.p_of_class(c, X)
        return class_probabilities

    def p_of_class(self, c, x):
        """ Pr(Y=c|x) """
        D = x.size

        class_probability = c['probability']
        feature_probabilities = np.ones(D)

        for i in range(D):
            if self.feature_types[i] == 'b':
                bernoulli = c['means'][i] if x[i] == 1 else 1 - c['means'][i]
                feature_probabilities[i] = bernoulli
            elif self.feature_types[i] == 'r':
                gauss = norm.pdf(x[i], c['means'][i], c['stds'][i])

                if gauss != 0:
                    feature_probabilities[i] = gauss
                else:
                    # ignore feature if standard deviation is zero
                    pass

        feature_probabilities = np.log(feature_probabilities)
        return np.exp(class_probability + np.sum(feature_probabilities))

    def create_classes(self, X, Y):
        N, __ = X.shape

        class_labels = np.unique(Y)
        # if len(class_labels) != self.num_classes:
        #     raise Exception('Number of classes does not match data.')

        classes = []

        for class_label in class_labels:
            idx = (Y == class_label)

            means = np.mean(X[idx], axis=0)
            stds = np.std(X[idx], axis=0)

            n_in_class, __ = X[idx].shape
            probability = np.log(float(n_in_class) / N)

            classes.append({'label': class_label,
                            'probability': probability,
                            'means': means, 'stds': stds})
        return classes
