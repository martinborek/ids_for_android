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

#datasets_to_compare = ["ngram2", "ngram3", "nostart/ngram2", "nostart/ngram3"]
#datasets_to_compare = ["ngram2", "ngram3", "20_ngram2", "20_ngram3", "20_ngram4", "40_ngram2", "40_ngram3"]
#datasets_to_compare = ["histogram", "20_histogram", "40_histogram"]

#datasets_to_compare = ["chunks/histogram"]
#datasets_to_compare = ["chunks_nostart/histogram", "chunks_nostart/ngram2", "chunks_nostart/ngram3"]
#datasets_titles = ["Chunks histogram nostart", "Chunks ngram2 nostart", "Chunks ngram3 nostart"]


#datasets_to_compare = ["chunks_nostart/com5", "chunks_nostart/com10"]
#datasets_titles = ["Chunks com5 nostart", "Chunks com10 nostart"]

#datasets_to_compare = ["com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9", "com10"]
#datasets_titles = ["Co-occurrence matrix 2 (all syscalls)", "Co-occurrence matrix 3 (all syscalls)", "Co-occurrence matrix 4 (all syscalls)", "Co-occurrence matrix 5 (all syscalls)",
#                   "Co-occurrence matrix 6 (all syscalls)", "Co-occurrence matrix 7 (all syscalls)", "Co-occurrence matrix 8 (all syscalls)", "Co-occurrence matrix 9 (all syscalls)", "Co-occurrence matrix 10 (all syscalls)"]


#datasets_to_compare = ["ngram2", "ngram3", "40_ngram2", "40_ngram3", "20_ngram2", "20_ngram3", "20_ngram4"]
#datasets_titles = ["N-gram 2 (all syscalls)", "N-gram 3 (all syscalls)", "N-gram 2 (40 syscalls)",  "N-gram 3 (40 syscalls)", "N-gram 2 (20 syscalls)",  "N-gram 3 (20 syscalls)",   "N-gram 4 (20 syscalls)"]
#datasets_to_compare = ["com5", "com10", "40_com5", "40_com10", "20_com5", "20_com10"]
#datasets_titles = ["Co-occurrence matrix 5 (all syscalls)", "Co-occurrence matrix 10 (all syscalls)", "Co-occurrence matrix 5 (40 syscalls)", "Co-occurrence matrix 10 (40 syscalls)",
#                    "Co-occurrence matrix 5 (20 syscalls)", "Co-occurrence matrix 10 (20 syscalls)",]

#datasets_to_compare = ["histogram", "40_histogram", "20_histogram"]
#datasets_titles = ["Histogram (all syscalls)", "Histogram (40 syscalls)", "Histogram (20 syscalls)"]

#params = [(0.1, 10), (0.1, 1), (0.1, 0.1), (0.1, 0.01), (0.1, 0.001), (0.01, 10), (0.01, 1), (0.01, 0.01), (0.01, 0.01), (0.01, 0.001), (0.3, 10), (0.3, 1), (0.3, 0.1), (0.3, 0.01), (0.3, 0.001), (0.001, 10), (0.001, 1), (0.001, 0.1), (0.3, 0.01), (0.001, 0.001) ]
#params = [(0.001, 0.001), (0.3, 10), (0.1, 10), (0.1, 0.001), (0.3, 100), (0.4, 100), (0.5, 100), (0.2, 100), (0.3, 1000)]
#params = [(0.4, 100), (0.2, 100), (0.01, 0.1)]


plt.figure(figsize=(11,9))

datasets_to_compare = ["20_histogram", "nostart/20_histogram"]
datasets_titles = ["Histogram (20 syscalls, original)", "Histogram (20 syscalls, no start)"]
for dataset_index in range(0, len(datasets_to_compare)):
#for nu, gamma in params:
    #dataset_index=0
    dataset = datasets_to_compare[dataset_index]

    benign_directory_path = "/home/mborekcz/MEGA/dataset/benign_feature_vectors/" + dataset + "/"
    malware_directory_path = "/home/mborekcz/MEGA/dataset/malware_feature_vectors/" + dataset + "/"

    benign_dataset_train = []
    benign_dataset_test = []


    defined_random_states = [233, 560, 5, 488, 178, 265, 891, 817, 963, 36] # COM
    # Pseudo random numbers that specify how the train/test data is split.
    # If train_test_split() called with the same number, the data is split
    # always in the same way. Can be used for debugging.
    random_states = []
    for i in range(0, 10):
        random_state = defined_random_states[i]
        # For a random state, uncommment the next line. We used exact states to be able to compare different methods
        #random_state = random.randint(1, 1000)

        random_states.append(random_state)

    i = 0
    for filename in os.listdir(benign_directory_path):
        file_path = os.path.join(benign_directory_path, filename)
        if os.path.isfile(file_path):
            random_state = random_states[i]
            i += 1

            loaded_data = np.loadtxt(file_path, delimiter=",")
            temp_train, temp_test = train_test_split(loaded_data, test_size=0.5, random_state=random_state)
            benign_dataset_train.extend(temp_train)
            benign_dataset_test.extend(temp_test)


    print("Benign data loaded, number of sets: {} {}".format(len(benign_dataset_train), len(benign_dataset_test)))


    #loaded_data = np.loadtxt('/home/mborekcz/MEGA/dataset/benign_feature_vectors/20_com3/x/com.onto.notepad.log', delimiter=",")
    #benign_dataset_test.extend(loaded_data)


    malware_dataset_test = []
    for filename in os.listdir(malware_directory_path):
        file_path = os.path.join(malware_directory_path, filename)
        if os.path.isfile(file_path):
            loaded_data = np.loadtxt(file_path, delimiter=",")
            malware_dataset_test.append(loaded_data)
    print("Malware data loaded, number of sets: {}".format(len(malware_dataset_test)))


    print("Pseudo-random numbers for splitting states: {}".format(random_states))

    print("Fitting the model...")
    #clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1) #ngram
    #clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.10) #com
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.01) #histogram

    clf.fit(benign_dataset_train)

    y_pred_train = clf.predict(benign_dataset_train)
    y_pred_benign_test = clf.predict(benign_dataset_test)
    y_pred_malware_test = clf.predict(malware_dataset_test)

    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_benign_test = y_pred_benign_test[y_pred_benign_test == -1].size
    n_error_malware_test = y_pred_malware_test[y_pred_malware_test == 1].size
    print("Results: errors train: {}/{} ; errors benign: {}/{} ; "
          "errors malware: {}/{}".format(n_error_train, y_pred_train.size,
                                               n_error_benign_test, y_pred_benign_test.size,
                                               n_error_malware_test, y_pred_malware_test.size))

    combined_dataset_test = np.r_[benign_dataset_test, malware_dataset_test]
    y_pred_combined = clf.decision_function(combined_dataset_test)
    y_true = np.array([1] * len(benign_dataset_test) + [0] * len(malware_dataset_test))

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_combined)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=datasets_titles[dataset_index] + ' [AUROC=%0.4f]' % roc_auc)


datasets_to_compare = ["ngram3", "nostart/ngram3"]
datasets_titles = ["N-gram 3 (all syscalls, original)", "N-gram 3 (all syscalls, no start)"]
for dataset_index in range(0, len(datasets_to_compare)):
#for nu, gamma in params:
    #dataset_index=0
    dataset = datasets_to_compare[dataset_index]

    benign_directory_path = "/home/mborekcz/MEGA/dataset/benign_feature_vectors/" + dataset + "/"
    malware_directory_path = "/home/mborekcz/MEGA/dataset/malware_feature_vectors/" + dataset + "/"

    benign_dataset_train = []
    benign_dataset_test = []


    defined_random_states = [233, 560, 5, 488, 178, 265, 891, 817, 963, 36] # COM
    # Pseudo random numbers that specify how the train/test data is split.
    # If train_test_split() called with the same number, the data is split
    # always in the same way. Can be used for debugging.
    random_states = []
    for i in range(0, 10):
        random_state = defined_random_states[i]
        # For a random state, uncommment the next line. We used exact states to be able to compare different methods
        #random_state = random.randint(1, 1000)

        random_states.append(random_state)

    i = 0
    for filename in os.listdir(benign_directory_path):
        file_path = os.path.join(benign_directory_path, filename)
        if os.path.isfile(file_path):
            random_state = random_states[i]
            i += 1

            loaded_data = np.loadtxt(file_path, delimiter=",")
            temp_train, temp_test = train_test_split(loaded_data, test_size=0.5, random_state=random_state)
            benign_dataset_train.extend(temp_train)
            benign_dataset_test.extend(temp_test)


    print("Benign data loaded, number of sets: {} {}".format(len(benign_dataset_train), len(benign_dataset_test)))


    #loaded_data = np.loadtxt('/home/mborekcz/MEGA/dataset/benign_feature_vectors/20_com3/x/com.onto.notepad.log', delimiter=",")
    #benign_dataset_test.extend(loaded_data)


    malware_dataset_test = []
    for filename in os.listdir(malware_directory_path):
        file_path = os.path.join(malware_directory_path, filename)
        if os.path.isfile(file_path):
            loaded_data = np.loadtxt(file_path, delimiter=",")
            malware_dataset_test.append(loaded_data)
    print("Malware data loaded, number of sets: {}".format(len(malware_dataset_test)))


    print("Pseudo-random numbers for splitting states: {}".format(random_states))

    print("Fitting the model...")

    clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1) #ngram
    #clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.10) #com
    #clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.01) #histogram


    clf.fit(benign_dataset_train)

    y_pred_train = clf.predict(benign_dataset_train)
    y_pred_benign_test = clf.predict(benign_dataset_test)
    y_pred_malware_test = clf.predict(malware_dataset_test)

    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_benign_test = y_pred_benign_test[y_pred_benign_test == -1].size
    n_error_malware_test = y_pred_malware_test[y_pred_malware_test == 1].size
    print("Results: errors train: {}/{} ; errors benign: {}/{} ; "
          "errors malware: {}/{}".format(n_error_train, y_pred_train.size,
                                               n_error_benign_test, y_pred_benign_test.size,
                                               n_error_malware_test, y_pred_malware_test.size))

    combined_dataset_test = np.r_[benign_dataset_test, malware_dataset_test]
    y_pred_combined = clf.decision_function(combined_dataset_test)
    y_true = np.array([1] * len(benign_dataset_test) + [0] * len(malware_dataset_test))

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_combined)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=datasets_titles[dataset_index] + ' [AUROC=%0.4f]' % roc_auc)

datasets_to_compare = ["com5", "nostart/com5"]
datasets_titles = ["Co-occurrence matrix 5 (all syscalls, original)", "Co-occurrence matrix 5 (all syscalls, no start)"]
for dataset_index in range(0, len(datasets_to_compare)):
#for nu, gamma in params:
    #dataset_index=0
    dataset = datasets_to_compare[dataset_index]

    benign_directory_path = "/home/mborekcz/MEGA/dataset/benign_feature_vectors/" + dataset + "/"
    malware_directory_path = "/home/mborekcz/MEGA/dataset/malware_feature_vectors/" + dataset + "/"

    benign_dataset_train = []
    benign_dataset_test = []


    defined_random_states = [233, 560, 5, 488, 178, 265, 891, 817, 963, 36] # COM
    # Pseudo random numbers that specify how the train/test data is split.
    # If train_test_split() called with the same number, the data is split
    # always in the same way. Can be used for debugging.
    random_states = []
    for i in range(0, 10):
        random_state = defined_random_states[i]
        # For a random state, uncommment the next line. We used exact states to be able to compare different methods
        #random_state = random.randint(1, 1000)

        random_states.append(random_state)

    i = 0
    for filename in os.listdir(benign_directory_path):
        file_path = os.path.join(benign_directory_path, filename)
        if os.path.isfile(file_path):
            random_state = random_states[i]
            i += 1

            loaded_data = np.loadtxt(file_path, delimiter=",")
            temp_train, temp_test = train_test_split(loaded_data, test_size=0.5, random_state=random_state)
            benign_dataset_train.extend(temp_train)
            benign_dataset_test.extend(temp_test)


    print("Benign data loaded, number of sets: {} {}".format(len(benign_dataset_train), len(benign_dataset_test)))


    #loaded_data = np.loadtxt('/home/mborekcz/MEGA/dataset/benign_feature_vectors/20_com3/x/com.onto.notepad.log', delimiter=",")
    #benign_dataset_test.extend(loaded_data)


    malware_dataset_test = []
    for filename in os.listdir(malware_directory_path):
        file_path = os.path.join(malware_directory_path, filename)
        if os.path.isfile(file_path):
            loaded_data = np.loadtxt(file_path, delimiter=",")
            malware_dataset_test.append(loaded_data)
    print("Malware data loaded, number of sets: {}".format(len(malware_dataset_test)))


    print("Pseudo-random numbers for splitting states: {}".format(random_states))

    print("Fitting the model...")


    #clf = svm.OneClassSVM(nu=0.01, kernel="rbf", gamma=0.1) #ngram
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=10) #com
    #clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.01) #histogram

    clf.fit(benign_dataset_train)

    y_pred_train = clf.predict(benign_dataset_train)
    y_pred_benign_test = clf.predict(benign_dataset_test)
    y_pred_malware_test = clf.predict(malware_dataset_test)

    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_benign_test = y_pred_benign_test[y_pred_benign_test == -1].size
    n_error_malware_test = y_pred_malware_test[y_pred_malware_test == 1].size
    print("Results: errors train: {}/{} ; errors benign: {}/{} ; "
          "errors malware: {}/{}".format(n_error_train, y_pred_train.size,
                                               n_error_benign_test, y_pred_benign_test.size,
                                               n_error_malware_test, y_pred_malware_test.size))

    combined_dataset_test = np.r_[benign_dataset_test, malware_dataset_test]
    y_pred_combined = clf.decision_function(combined_dataset_test)
    y_true = np.array([1] * len(benign_dataset_test) + [0] * len(malware_dataset_test))

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_combined)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=datasets_titles[dataset_index] + ' [AUROC=%0.4f]' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
