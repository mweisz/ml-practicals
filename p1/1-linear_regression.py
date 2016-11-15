import cPickle as cp
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso

# Using seaborn as default style for matplotlib
import seaborn
seaborn.set()


def fit_linear_model(X, Y):
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)),
               np.transpose(X)), Y)
    return w


def compute_mean_squared_error(Y, Y_pred):
    return (1.0 / len(Y)) * np.dot(Y - Y_pred, Y - Y_pred)


def normalise(X, mean=None, std=None):
    if (mean is None) or (std is None):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

    X_norm = (mean - X) / std
    return X_norm, mean, std


def perform_regression(clf, degree, X_train, Y_train, X_test, Y_test):
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_expanded = poly.fit_transform(X_train)
    X_test_expanded = poly.fit_transform(X_test)

    # Normalise expanded data
    X_train_expanded_norm, train_mean, train_std = normalise(X_train_expanded)
    X_test_expanded_norm, __, __ = normalise(X_test_expanded, mean=train_mean, std=train_std)

    # Add Ones Column for bias term
    N_train, __ = X_train.shape
    N_test, __ = X_test.shape
    X_train_expanded_norm = np.column_stack((np.ones(N_train), X_train_expanded_norm))
    X_test_expanded_norm = np.column_stack((np.ones(N_test), X_test_expanded_norm))

    clf.fit(X_train_expanded_norm, Y_train)
    y_pred = clf.predict(X_test_expanded_norm)
    mse = compute_mean_squared_error(Y_test, y_pred)

    return mse


# Loading the data
X, y = cp.load(open('data/winequality-white.cPickle', 'rb'))

N, D = X.shape
N_train = int(0.8 * N)
N_test = N - N_train
X_train = X[:N_train]
y_train = y[:N_train]
X_test = X[N_train:]
y_test = y[N_train:]

# Plot the dist. of target variable Y (wine ratings) in the training set
MIN_RATING = 0
MAX_RATING = 10
y_range = range(MIN_RATING, MAX_RATING + 1)
frequencies, __ = np.histogram(y_train, bins=len(y_range),
                               range=(y_range[0], y_range[-1]))
plt.bar(y_range, frequencies, align='center')
plt.xticks(y_range, y_range)
plt.xlabel('Quality Rating')
plt.ylabel('N')
plt.title('Distribution of Quality Ratings for White Wines')
plt.show()

# Use the average y value in test set as predictor
y_train_mean = np.mean(y_train)

# Mean squared error in training set
y_train_mse_naive = compute_mean_squared_error(y_train, y_train_mean)
# Mean squared error in test set
y_test_mse_naive = compute_mean_squared_error(y_test, y_train_mean)

print "Using average quality rating in training set as predictor:"
print "\tMean sq. error in training set: %s." % (y_train_mse_naive)
print "\tMean sq. error in test set: %s." % (y_test_mse_naive)


# Normalise the data
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)
X_train_norm_without_ones = (X_train_mean - X_train) / X_train_std
X_test_norm_without_ones = (X_train_mean - X_test) / X_train_std

# Add Ones Column for bias term
X_train_norm = np.column_stack((np.ones(N_train), X_train_norm_without_ones))
X_test_norm = np.column_stack((np.ones(N_test), X_test_norm_without_ones))

w = fit_linear_model(X_train_norm, y_train)

y_train_pred = np.dot(X_train_norm, w)
y_train_mse_linear_model = compute_mean_squared_error(y_train, y_train_pred)
y_test_pred = np.dot(X_test_norm, w)
y_test_mse_linear_model = compute_mean_squared_error(y_test, y_test_pred)

print "Using linear regression as predictor:"
print "\tMean sq. error in training set: %s." % (y_train_mse_linear_model)
print "\tMean sq. error in test set: %s." % (y_test_mse_linear_model)


# Compute learning curve
mse_train = []
mse_test = []
learning_curve_range = range(20, 620, 20)
for k in learning_curve_range:
    w = fit_linear_model(X_train_norm[0:k], y_train[0:k])
    Y_train_pred = np.dot(X_train_norm[0:k], w)
    Y_test_pred = np.dot(X_test_norm, w)
    mse_train.append(compute_mean_squared_error(y_train[0:k], Y_train_pred))
    mse_test.append(compute_mean_squared_error(y_test, Y_test_pred))

# Plot learning curve
plt.title('Learning Curve')
plt.ylabel('Mean squared error')
plt.plot(learning_curve_range, mse_train, label="Training Error")
plt.plot(learning_curve_range, mse_test, label="Test Error")
plt.legend()
plt.show()


# Potential params for Ridge Regression and Lasso
MIN_EXPONENT = -5
MAX_EXPONENT = 5
lambdas = [np.power(10, float(x)) for x in range(MIN_EXPONENT, MAX_EXPONENT+1)]
degrees = [2, 3, 4]

# Split training set into 'smaller training set' and validation set
N_val = int(0.2 * N_train)
N_train_without_val = N_train - N_val

X_train_without_val = X_train[:N_train_without_val]
# X_train_without_val = X_train_norm_without_ones[:N_train_without_val]
y_train_without_val = y_train[:N_train_without_val]

X_val = X_train[N_train_without_val:]
y_val = y_train[N_train_without_val:]


# Ridge Regression: Init params
best_lambda_ridge = lambdas[0]
best_degree_ridge = degrees[0]
min_error_ridge = float('inf')

# Lasso: Init params
best_lambda_lasso = lambdas[0]
best_degree_lasso = degrees[0]
min_error_lasso = float('inf')

print("Finding hyper-parameters for Ridge and Lasso: ")
for d in degrees:
    for l in lambdas:
        # Ridge regression
        ridge_mse = perform_regression(Ridge(alpha=l), d,
                                       X_train_without_val, y_train_without_val,
                                       X_val, y_val)

        # Keep track of best hyper params
        if ridge_mse <= min_error_ridge:
            print "\tRidge (%s,%s) => %s (MSE)" % (l, d, ridge_mse)
            min_error_ridge = ridge_mse
            best_lambda_ridge = l
            best_degree_ridge = d

        # Lasso regression
        lasso_mse = perform_regression(Lasso(alpha=l), d,
                                       X_train_without_val, y_train_without_val,
                                       X_val, y_val)

        # Keep track of best hyper params
        if lasso_mse <= min_error_lasso:
            print "\tLasso (%s,%s) => %s (MSE)" % (l, d, lasso_mse)
            min_error_lasso = lasso_mse
            best_lambda_lasso = l
            best_degree_lasso = d


# Train model using entire training set with optimal hyper params:
#
# Ridge
ridge_mse = perform_regression(Ridge(alpha=best_lambda_ridge), best_degree_ridge,
                               X_train, y_train,
                               X_test, y_test)

print "Using ridge regression with (lambda=%s, d=%s) as predictor:" \
    % (best_lambda_ridge, best_degree_ridge)
print "\tMean sq. error in test set: %s." % (ridge_mse)

#
# Lasso
lasso_mse = perform_regression(Lasso(alpha=best_lambda_lasso), best_degree_lasso,
                               X_train, y_train,
                               X_test, y_test)

print "Using lasso regression with (lambda=%s, d=%s) as predictor:" \
    % (best_lambda_lasso, best_degree_lasso)
print "\tMean sq. error in test set: %s." % (lasso_mse)
