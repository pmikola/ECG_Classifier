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
import umap
from model import ECGClassifier

matplotlib.use('TkAgg')

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, num_classes=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if num_classes is not None:
            self.class_weights = nn.Parameter(torch.ones(num_classes, device=torch.device('cuda')))
        else:
            self.class_weights = None
    def forward(self, inputs, targets):
        if self.class_weights is not None:
            weight = torch.softmax(self.class_weights, dim=0)
        else:
            weight = None
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1 - pt) ** self.gamma
        if weight is not None:
            sample_weight = weight[targets]
        else:
            sample_weight = 1.0
        loss = - sample_weight * focal_factor * log_pt
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

device = torch.device('cuda')
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

reducer = umap.UMAP(
    n_neighbors=10,
    n_components=2,
    min_dist=0.99,
    spread=2.0,
    set_op_mix_ratio=1.0,
    local_connectivity=1,
    repulsion_strength=15.0,
    negative_sample_rate=10,
    n_epochs=5000,
    learning_rate=1,
    init='spectral',
    random_state=42,
    metric='euclidean',
    verbose=True
)
X_umap = reducer.fit_transform(X_train.reshape(X_train.shape[0], -1))
plt.figure(figsize=(8,6), facecolor='black')
ax = plt.gca()
ax.set_facecolor('black')
colors = plt.get_cmap("Set1").colors
for c in range(num_classes):
    idx = np.where(train_int_labels.cpu().numpy() == c)[0]
    plt.scatter(X_umap[idx,0], X_umap[idx,1], color=colors[c % len(colors)], label=str(c), s=5)
plt.title("UMAP Visualization of Training Data", color='white')
plt.legend(facecolor='black', edgecolor='white', labelcolor='white')
ax.tick_params(colors='white')
plt.show()


num_epochs = 10000
batch_size = 32
lead_view = 0
seq_len = 1000
model = ECGClassifier(num_classes, seq_len).to(device)
print('Model Parameters: ', count_parameters(model))
class_weights = torch.ones(num_classes, device=device)
class_weights[2] = 2.0
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-6, amsgrad=True)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

train_loss_history = []
test_loss_history  = []
train_acc_history  = []
test_acc_history   = []
start_block_time = time.time()

def augment_data(data):
    if data.ndim == 2:
        data = data.unsqueeze(1)
    aug = torch.randint(0, 4, (1,), device=device).item()
    if aug == 0:
        noise_std = 0.01
        noise = torch.randn_like(data) * noise_std
        return data + noise
    elif aug == 1:
        shift = torch.randint(-10, 10, (1,), device=device).item()
        return data.roll(shift, dims=-1)
    elif aug == 2:
        scale = torch.empty(1, device=device).uniform_(0.9, 1.1).item()
        return data * scale
    elif aug == 3:
        noise_std = 0.01
        noise = torch.randn_like(data, device=device) * noise_std
        data_noisy = data + noise
        num_segments = 4
        b, c, l = data_noisy.shape
        segment_length = l // num_segments
        if segment_length > 0:
            segments = []
            for i in range(num_segments):
                start = i * segment_length
                end = start + segment_length if i < num_segments - 1 else l
                segments.append(data_noisy[:, :, start:end])
            perm = torch.randperm(num_segments, device=device)
            return torch.cat([segments[i] for i in perm], dim=2)
        else:
            return data_noisy

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
    train_data_sample = augment_data(train_data_sample)
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
        print(f"[{epoch:04d}/{num_epochs}] Train Loss: {train_loss.item():.4f} | Test Loss: {test_loss.item():.4f} | Train Acc: {train_acc:.2f} | Test Acc: {test_acc:.2f} | Mean Time/Epoch: {mean_block_time:.3f}s")
        start_block_time = time.time()
    scheduler.step()

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
fig.suptitle("ECG Classifier Results", fontsize=16)
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
cm_norm = 100 * cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)
plt.figure(figsize=(8, 6))
plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (%)')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, [str(i) for i in range(num_classes)])
plt.yticks(tick_marks, [str(i) for i in range(num_classes)])
plt.ylabel('True label')
plt.xlabel('Predicted label')
thresh = cm_norm.max() / 2.
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        plt.text(j, i, f"{cm_norm[i, j]:.1f}%", horizontalalignment="center", color="white" if cm_norm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()
