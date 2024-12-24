import time
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

program_start_time = time.time()

# Φόρτωση των δεδομένων CIFAR-10
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Συνάρτηση για τη φόρτωση της CIFAR-10
def load_cifar10_data(folder_path):
    train_data = None
    train_labels = []
    for i in range(1, 6):
        batch = unpickle(f"{folder_path}/data_batch_{i}")
        batch_data = batch[b'data']
        if train_data is None:
            train_data = batch_data
        else:
            train_data = np.concatenate((train_data, batch_data), axis=0)
        train_labels.extend(batch[b'labels'])

    test_batch = unpickle(f"{folder_path}/test_batch")
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']

    # Φόρτωση των metadata για τα labels
    meta_data = unpickle(f"{folder_path}/batches.meta")
    label_names = meta_data[b'label_names']
    label_names = [label.decode('utf-8') for label in label_names]  # Μετατροπή των ετικετών από bytes σε string

    # Μετατροπή labels σε NumPy arrays
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Τυποποίηση δεδομένων
    sc = StandardScaler()
    train_data = sc.fit_transform(train_data)
    test_data = sc.transform(test_data)

    # Εφαρμογή PCA για διατήρηση 95% της πληροφορίας
    pca_st = time.time()
    pca = PCA(0.95)
    train_data = pca.fit_transform(train_data)
    test_data = pca.transform(test_data)
    pca_end = time.time()
    print(f"Διαστάσεις μετά το PCA: {train_data.shape[1]}, σε {pca_end - pca_st:.2f} δευτερόλεπτα")

    return train_data, train_labels, test_data, test_labels, label_names

# Καθορισμός της διαδρομής του φακέλου
folder_path = "C:/Users/gouti/Downloads/cifar-10-python/cifar-10-batches-py"
train_data, train_labels, test_data, test_labels, label_names = load_cifar10_data(folder_path)

# Ορισμός αριθμού δειγμάτων ανά κατηγορία (Sampling)
"""
samples_per_class_train = 1000
samples_per_class_test = 150

for i in range(10):
   idx = []
   for k in range(len(train_labels)):
      if train_labels[k] == i:
        idx.append(k)
   sample_idx = np.random.choice(idx, samples_per_class_train, replace=False)
   if i == 0:
       train_data_sampled = train_data[sample_idx]
       train_labels_sampled = train_labels[sample_idx]
   else:
       train_data_sampled = np.concatenate((train_data_sampled, train_data[sample_idx]), axis=0)
       train_labels_sampled = np.concatenate((train_labels_sampled, train_labels[sample_idx]), axis=0)

for i in range(10):
   idx = []
   for k in range(len(test_labels)):
      if test_labels[k] == i:
        idx.append(k)
   sample_idx = np.random.choice(idx, samples_per_class_test, replace=False)
   if i == 0:
       test_data_sampled = test_data[sample_idx]
       test_labels_sampled = test_labels[sample_idx]
   else:
       test_data_sampled = np.concatenate((test_data_sampled, test_data[sample_idx]), axis=0)
       test_labels_sampled = np.concatenate((test_labels_sampled, test_labels[sample_idx]), axis=0)
"""

# Υπολογισμός ακρίβειας ανά κατηγορία
def accuracy_per_category(test_labels, predicted, label_names):
    class_correct = []
    class_total = []

    for i in range(10):
        class_correct.append(0)
        class_total.append(0)

    for i in range(len(test_labels)):
        label = test_labels[i].item()
        class_total[label] += 1
        if predicted[i].item() == label:
            class_correct[label] += 1

    for i in range(10):
        if class_total[i] > 0:
          accuracy = 100 * class_correct[i] / class_total[i]
        else:
            accuracy = 0
        print(f"Κατηγορία: {label_names[i]:<10s} | Σωστά: {class_correct[i]:<3} / {class_total[i]:<3} | Ακρίβεια: {accuracy:.2f}%")


# Διαφορετικές τιμές C, gamma, decision_function_shape
C_values = [0.1, 0.5, 1.0, 10.0]
gammas = [0.0002, 0.0005, 0.001, 0.005]
dfs = ['ovr', 'ovo']
i = 0

for vs in dfs:
    for C in C_values:
        for gamma in gammas:
            i += 1
            print(f"\n Συνδυασμός {i}, Εκπαίδευση SVM με πυρήνα RBF, C={C}, gamma={gamma} και decision_function_shape={vs}")

            # Εκπαίδευση
            start_time = time.time()
            model = SVC(kernel='rbf', C=C, gamma=gamma, decision_function_shape=vs)
            model.fit(train_data, train_labels)
            train_time = time.time() - start_time

            # Αξιολόγηση
            start_time = time.time()
            predictions_train = model.predict(train_data)
            predictions_test = model.predict(test_data)
            train_accuracy = accuracy_score(train_labels, predictions_train) * 100
            test_accuracy = accuracy_score(test_labels, predictions_test) * 100
            pred_time = time.time() - start_time

            print(
                f"Πυρήνας: RBF, C={C}, gamma={gamma} και decision_function_shape={vs}, "f"Ακρίβεια στο Training set: {train_accuracy:.2f}%, "f"Ακρίβεια στο Test set: {test_accuracy:.2f}%, "f"Χρόνος Εκπαίδευσης: {train_time/60:.2f} λεπτά, "f"Χρόνος Προβλέψεων: {pred_time/60:.2f} λεπτά")

            accuracy_per_category(test_labels, predictions_test, label_names)


program_end_time = time.time()
print(f"Χρόνος εκτέλεσης προγράμματος: {(program_end_time-program_start_time)/3600:.2f} ώρες")