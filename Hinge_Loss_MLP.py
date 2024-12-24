import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

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

    train_data, train_labels = torch.tensor(train_data).float(), torch.tensor(train_labels)
    test_data, test_labels = torch.tensor(test_data).float(), torch.tensor(test_labels)

    return train_data, train_labels, test_data, test_labels, label_names

# Καθορισμός της διαδρομής του φακέλου
folder_path = "C:/Users/gouti/Downloads/cifar-10-python/cifar-10-batches-py"
train_data, train_labels, test_data, test_labels, label_names = load_cifar10_data(folder_path)

# Δημιουργία DataLoader με batch size
batch_size = 1024  # Μεγαλύτερο batch size για σταθερά gradients
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Ορισμός του MLP με ένα κρυφό επίπεδο
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(train_data.shape[1], 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Το τελικό επίπεδο χωρίς ReLU
        return x


# Ορισμός μοντέλου, απώλειας και βελτιστοποιητή
model = MLP()
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

#Κώδικας για hinge loss βελτιστοποίηση
def hinge_loss(outputs, labels):
    sum_all = []
    for sample, correct_label in zip(outputs, labels):
        sum_sample = 0
        correct_label_item = sample[correct_label]
        for i in range(10):
            if i != correct_label:
                sum_sample += max(0, sample[i] - correct_label_item + 1)
        sum_all.append(sum_sample)
    loss = sum(sum_all) / len(sum_all)
    return loss

# Early Stopping parameters
patience = 20
best_loss = 1000.0
counter = 0

# Λίστα για αποθήκευση του loss σε επιλεγμένες εποχές
selected_epochs = []
selected_losses = []

prev_lr = optimizer.param_groups[0]['lr']

# Εκπαίδευση MLP
model.train()
num_epochs = 1000
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    # Προβλέψεις και υπολογισμός του hinge loss
    for batch_data, batch_labels in train_loader:
        outputs = model(batch_data)
        loss = hinge_loss(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    # Ελέγχω αν το learning rate άλλαξε
    current_lr = optimizer.param_groups[0]['lr']
    if current_lr != prev_lr:
        print(f"Το learning rate άλλαξε από {prev_lr:.6f} σε {current_lr:.6f} στο τέλος της εποχής {epoch}")
        prev_lr = current_lr

    if (epoch + 1) <= 20 or (epoch + 1) % 50 == 0:
        selected_epochs.append(epoch + 1)
        selected_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    scheduler.step(avg_loss)

    # Early Stopping Check
    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            if (epoch + 1) not in selected_epochs:
                selected_epochs.append(epoch + 1)
                selected_losses.append(avg_loss)
            print(f"Early stopping at epoch {epoch + 1}, Loss: {avg_loss:.4f}")
            break

end_time = time.time()
print(f"Χρόνος εκπαίδευσης: {(end_time - start_time) / 60:.2f} λεπτά")

# Δημιουργία διαγράμματος για το loss
plt.figure(figsize=(10, 6))
plt.plot(selected_epochs, selected_losses, label="Selected Loss", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss σε Επιλεγμένες Εποχές")
plt.legend()
plt.grid()
plt.show()

# Υπολογισμός ακρίβειας
model.eval()

def predict_acc(data, labels):
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)  # Παίρνουμε την κατηγορία με τη μεγαλύτερη πιθανότητα
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct / labels.size(0)
    return accuracy, predicted


train_accuracy_mlp, predicted_train = predict_acc(train_data, train_labels)
test_accuracy_mlp, predicted_test = predict_acc(test_data, test_labels)

print(f"\nMLP - Ακρίβεια Εκπαίδευσης: {train_accuracy_mlp:.2f}%")
print(f"MLP - Ακρίβεια Ελέγχου: {test_accuracy_mlp:.2f}%")

#Υπολογισμός ακρίβειας ανά κατηγορία
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

accuracy_per_category(test_labels, predicted_test, label_names)

program_end_time = time.time()
print(f"Χρόνος εκτέλεσης προγράμματος: {(program_end_time-program_start_time)/60:.2f} λεπτά")