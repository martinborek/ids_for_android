import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
import random

f1sc = make_scorer(f1_score)

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# Generate train data
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
#####################^^^^^^^^^^^^^^^^^^^^^^^^^^^ DATA GENERATION

print("Loading data")
directory_path = "/home/mborekcz/MEGA/dataset/feature_vectors/com5/"
dataset_train = []
dataset_test = []

# Pseudo random numbers that specify how the train/test data is split.
# If train_test_split() called with the same number, the data is split
# always in the same way. Can be used for debugging.
random_states = []
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if os.path.isfile(file_path):
        random_state = random.randint(1, 1000)
        random_states.append(random_state)
        loaded_data = np.loadtxt(file_path, delimiter=",")
        temp_train, temp_test = train_test_split(loaded_data, test_size=0.25, random_state=random_state)
        dataset_train.extend(temp_train)
        dataset_test.extend(temp_test)


print("Data loaded, number of sets: {} {}".format(len(dataset_train), len(dataset_test)))
print("Pseudo-random numbers for splitting states: {}".format(random_states))

print("Fitting the model...")
svm_parameters_grid = {'nu': [0.05, 0.1, 0.5], 'gamma': [0.001, 0.01, 0.1, 0.5]}
clf = GridSearchCV(svm.OneClassSVM(kernel="rbf"), param_grid=svm_parameters_grid)
#svm_parameters_grid = {'kernel': ['rbf'], 'nu': [0.01, 0.05, 0.1, 0.5], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]}
#clf = GridSearchCV(svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1), svm_parameters_grid, scoring=f1sc)
#clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(dataset_train)
print("DONE")
exit(0)

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size


########################VVVVVVVVVVVVVVVVVVVVVVVVVVV PLOTTING
# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s)
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s)
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s)
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "error train: %d/200 ; errors novel regular: %d/40 ; "
    "errors novel abnormal: %d/40"
    % (n_error_train, n_error_test, n_error_outliers))
plt.show()
