import ast
import sys
import time

import matplotlib
import wfdb
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.cuda.amp as amp
from model import EcgFNOClassifier

matplotlib.use('TkAgg')

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='mean')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

def one_hot_encode(labels, num_classes):
    vector = np.zeros(num_classes, dtype=np.float32)
    for label in labels:
        vector[class_to_idx[label]] = 1.0
    return vector

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

path = 'ptb-xl-dataset/'
sampling_rate = 100

Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

X = load_raw_data(Y, sampling_rate, path)

agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]


Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

test_fold = 10
X_train = X[np.where(Y.strat_fold != test_fold)]
X_test = X[np.where(Y.strat_fold == test_fold)]

y_train = np.array(Y[(Y.strat_fold != test_fold)].diagnostic_superclass)
y_test  = np.array(Y[Y.strat_fold == test_fold].diagnostic_superclass)

unique_classes = np.unique(np.concatenate(Y['diagnostic_superclass'].values))
print(unique_classes)
num_classes = len(unique_classes)
class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}

X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=device)

y_train_ohe = np.array([one_hot_encode(labels, num_classes) for labels in y_train])
y_test_ohe  = np.array([one_hot_encode(labels, num_classes) for labels in y_test])

y_train_t = torch.tensor(y_train_ohe, dtype=torch.float32, device=device)
y_test_t  = torch.tensor(y_test_ohe,  dtype=torch.float32, device=device)

train_int_labels = torch.argmax(y_train_t, dim=1)
indices_by_class = {c: (train_int_labels == c).nonzero(as_tuple=True)[0] for c in range(num_classes)}

num_epochs = 10000
batch_size = 64
lead_view = 0
modes = 5
hidden_width = 200
number_of_points = 1000
model = EcgFNOClassifier(number_of_points, modes, hidden_width, num_classes).to(device)
print('Model Parameters: ', count_parameters(model))
criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss(gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5, amsgrad=True)

train_loss_history = []
test_loss_history  = []
train_acc_history  = []
test_acc_history   = []

start_block_time = time.time()

for epoch in range(num_epochs):
    samples_per_class = batch_size // num_classes
    remainder = batch_size % num_classes
    balanced_indices = []
    for c in range(num_classes):
        idx = indices_by_class[c]
        chosen = idx[torch.randint(0, idx.shape[0], (samples_per_class,), device=device)]
        balanced_indices.append(chosen)
    for i in range(remainder):
        c = i % num_classes
        idx = indices_by_class[c]
        chosen = idx[torch.randint(0, idx.shape[0], (1,), device=device)]
        balanced_indices.append(chosen)
    random_train_indices = torch.cat(balanced_indices)
    random_train_indices = random_train_indices[torch.randperm(random_train_indices.shape[0], device=device)]
    random_test_indices = torch.randperm(X_test_t.shape[0], device=device)[:batch_size]

    train_data_sample = X_train_t[random_train_indices, :, lead_view]
    train_label_sample = y_train_t[random_train_indices]

    test_data_sample = X_test_t[random_test_indices, :, lead_view]
    test_label_sample = y_test_t[random_test_indices]

    model.train()
    train_outputs = model(train_data_sample)
    train_loss = criterion(train_outputs, torch.argmax(train_label_sample, dim=1))

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    train_label_indices = torch.argmax(train_label_sample, dim=1)
    _, train_preds = torch.max(train_outputs, dim=1)
    train_correct = (train_preds == train_label_indices).sum().item()
    train_acc = train_correct / train_label_indices.size(0)

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data_sample)
        test_loss = criterion(test_outputs, torch.argmax(test_label_sample, dim=1))

    test_label_indices = torch.argmax(test_label_sample, dim=1)
    _, test_preds = torch.max(test_outputs, dim=1)
    test_correct = (test_preds == test_label_indices).sum().item()
    test_acc = test_correct / test_label_indices.size(0)

    if epoch % 10 == 0:
        train_loss_history.append(train_loss.item())
        test_loss_history.append(test_loss.item())
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

    if epoch % 50 == 0 and epoch > 0:
        end_block_time = time.time()
        block_elapsed = end_block_time - start_block_time
        mean_block_time = block_elapsed / 50.0
        print(
            f"[{epoch:04d}/{num_epochs}] "
            f"Train Loss: {train_loss.item():.4f} | "
            f"Test Loss: {test_loss.item():.4f} | "
            f"Train Acc: {train_acc:.2f} | Test Acc: {test_acc:.2f} | "
            f"Mean Time/Epoch: {mean_block_time:.3f}s")
        start_block_time = time.time()

plt.style.use('dark_background')

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

axs[0].plot(train_loss_history, label='Train Loss')
axs[0].plot(test_loss_history, label='Test Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training and Test Loss')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(train_acc_history, label='Train Accuracy')
axs[1].plot(test_acc_history, label='Test Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Training and Test Accuracy')
axs[1].legend()
axs[1].grid(True)

fig.suptitle("ECG FNO Classifier Results", fontsize=16)
plt.tight_layout()
plt.show()

model.eval()
no_samples_conf_matrix = 1000
with torch.no_grad():
    random_test_indices = torch.randperm(X_test_t.shape[0], device=device)[:no_samples_conf_matrix]
    test_data_sample = X_test_t[random_test_indices, :, lead_view]
    test_label_sample = y_test_t[random_test_indices]
    outputs_full = model(test_data_sample)
    _, preds_full = torch.max(outputs_full, dim=1)
    test_label_indices = torch.argmax(test_label_sample, dim=1)

cm = confusion_matrix(test_label_indices.cpu().numpy(), preds_full.cpu().numpy())

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, [str(i) for i in range(num_classes)])
plt.yticks(tick_marks, [str(i) for i in range(num_classes)])
plt.ylabel('True label')
plt.xlabel('Predicted label')

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()
