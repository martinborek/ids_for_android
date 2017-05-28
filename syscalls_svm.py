import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import f1_score, make_scorer
import random
#xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
print("Loading data")

benign_directory_path = "/home/mborekcz/MEGA/dataset/benign_feature_vectors/com5/"
benign_dataset_train = []
benign_dataset_test = []

# Pseudo random numbers that specify how the train/test data is split.
# If train_test_split() called with the same number, the data is split
# always in the same way. Can be used for debugging.
random_states = []
for filename in os.listdir(benign_directory_path):
    file_path = os.path.join(benign_directory_path, filename)
    if os.path.isfile(file_path):
        random_state = random.randint(1, 1000)
        random_states.append(random_state)
        loaded_data = np.loadtxt(file_path, delimiter=",")
        temp_train, temp_test = train_test_split(loaded_data, test_size=0.4, random_state=random_state)
        benign_dataset_train.extend(temp_train)
        benign_dataset_test.extend(temp_test)


print("Benign data loaded, number of sets: {} {}".format(len(benign_dataset_train), len(benign_dataset_test)))


malware_directory_path = "/home/mborekcz/MEGA/dataset/malware_feature_vectors/com5/"
malware_dataset_test = []
for filename in os.listdir(malware_directory_path):
    file_path = os.path.join(malware_directory_path, filename)
    if os.path.isfile(file_path):
        loaded_data = np.loadtxt(file_path, delimiter=",")
        malware_dataset_test.append(loaded_data)
print("Malware data loaded, number of sets: {}".format(len(malware_dataset_test)))


print("Pseudo-random numbers for splitting states: {}".format(random_states))

print("Fitting the model...")
#svm_parameters_grid = {'nu': [0.05, 0.1, 0.5], 'gamma': [0.001, 0.01, 0.1, 0.5]}
#clf = GridSearchCV(svm.OneClassSVM(kernel="rbf"), param_grid=svm_parameters_grid)
#svm_parameters_grid = {'kernel': ['rbf'], 'nu': [0.01, 0.05, 0.1, 0.5], 'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]}
#clf = GridSearchCV(svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1), svm_parameters_grid, scoring=f1sc)
#clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1)
clf = svm.OneClassSVM(kernel="rbf")

clf.fit(benign_dataset_train)

y_pred_train = clf.predict(benign_dataset_train)
y_pred_benign_test = clf.predict(benign_dataset_test)
y_pred_malware_test = clf.predict(malware_dataset_test)


n_error_train = y_pred_train[y_pred_train == -1].size
n_error_benign_test = y_pred_benign_test[y_pred_benign_test == -1].size
n_error_malware_test = y_pred_malware_test[y_pred_malware_test == 1].size
print("Results: error train: {}/{} ; errors novel benign: {}/{} ;"
      "errors novel malware: {}/{}".format(n_error_train, y_pred_train.size,
                                           n_error_benign_test, y_pred_benign_test.size,
                                           n_error_malware_test, y_pred_malware_test.size))

combined_dataset_test = np.r_[benign_dataset_test, malware_dataset_test]
y_pred_combined = clf.predict(combined_dataset_test)
y_true = np.array([1] * len(benign_dataset_test) + [0] * len(malware_dataset_test))

fpr, tpr, thresholds = roc_curve(y_true, y_pred_combined)
roc_auc = auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

exit()

########################VVVVVVVVVVVVVVVVVVVVVVVVVVV PLOTTING
# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(benign_dataset_train[:, 0], benign_dataset_train[:, 1], c='white', s=s)
b2 = plt.scatter(benign_dataset_test[:, 0], benign_dataset_test[:, 1], c='blueviolet', s=s)
c = plt.scatter(malware_dataset_test[:, 0], malware_dataset_test[:, 1], c='gold', s=s)
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
