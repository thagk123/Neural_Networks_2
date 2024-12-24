import time
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
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

# Διαφορετικές τιμές για το C
C_values = [0.1, 0.5, 1.0, 10.0]

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


for C in C_values:
    print(f"\nΕκπαίδευση SVM με πυρήνα Linear και C={C}")

    # Εκπαίδευση
    start_time = time.time()
    model = LinearSVC(C=C)
    model.fit(train_data, train_labels)
    train_time = time.time() - start_time

    # Αξιολόγηση
    start_time = time.time()
    predictions_train = model.predict(train_data)
    predictions_test = model.predict(test_data)
    train_accuracy = accuracy_score(train_labels, predictions_train) * 100
    test_accuracy = accuracy_score(test_labels, predictions_test) * 100
    pred_time = time.time() - start_time

    print(f"Πυρήνας: Linear, C={C}, "f"Ακρίβεια στο Training set: {train_accuracy:.2f}%, "f"Ακρίβεια στο Test set: {test_accuracy:.2f}%, "f"Χρόνος Εκπαίδευσης: {train_time:.2f} δευτερόλεπτα, "f"Χρόνος Προβλέψεων: {pred_time:.2f} δευτερόλεπτα")

    accuracy_per_category(test_labels, predictions_test, label_names)


program_end_time = time.time()
print(f"Χρόνος εκτέλεσης προγράμματος: {(program_end_time-program_start_time)/60:.2f} λεπτά")