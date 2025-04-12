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
import pickle
from model import ECGClassifier

matplotlib.use('TkAgg')

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if class_weight is not None:
            self.class_weights = nn.Parameter(class_weight)
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

def one_hot_cross_entropy_loss(logits, target_one_hot):
    log_probs = F.log_softmax(logits, dim=1)
    loss = - torch.sum(target_one_hot * log_probs, dim=1)
    return loss.mean()

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

def ecg_umap_plot(X_train):
    reducer = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.99, spread=2.0,
                        set_op_mix_ratio=1.0, local_connectivity=1, repulsion_strength=15.0,
                        negative_sample_rate=10, n_epochs=5000, learning_rate=1,
                        init='spectral', random_state=42, metric='euclidean', verbose=True)
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

def augment_data(data):
    if data.ndim == 2:
        data = data.unsqueeze(1)
    aug = torch.randint(0, 5, (1,), device=device).item()
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
        return data
    else:
        noise_std = 0.01
        noise = torch.randn_like(data, device=device) * noise_std
        data_noisy = data + noise
        b, c, l = data_noisy.shape
        num_segments = 4
        seg_len = l // num_segments
        if seg_len > 0:
            segments = []
            for i in range(num_segments):
                start = i * seg_len
                end = start + seg_len if i < num_segments - 1 else l
                segments.append(data_noisy[:, :, start:end])
            perm = torch.randperm(num_segments, device=data_noisy.device)
            return torch.cat([segments[i] for i in perm], dim=2)
        else:
            return data_noisy

def fuse_logits_dynamic(main_logits, aux_positive):
    with torch.no_grad():
        main_probs = F.softmax(main_logits, dim=1)
        main_conf, _ = main_probs.max(dim=1)
        aux_conf = torch.abs(aux_positive - 0.5) * 2
        conf_sum = main_conf + aux_conf + 1e-8
        w_main = (main_conf / conf_sum).unsqueeze(1)
        w_aux = (aux_conf / conf_sum).unsqueeze(1)
    fused = main_logits.clone()
    combined_2 = w_main * fused[:, 2:3] + w_aux * aux_positive.unsqueeze(1)
    fused_logits = torch.cat([fused[:, :2], combined_2, fused[:, 3:]], dim=1)
    return fused_logits

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
    def forward(self, features, labels):
        centers_batch = self.centers[labels]
        loss = F.mse_loss(features, centers_batch)
        return loss

device = torch.device('cuda')
print("Using device:", device)

dataset_path = "dataset.pkl"
if os.path.exists(dataset_path):
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    X_train_t = dataset["X_train_t"]
    X_test_t = dataset["X_test_t"]
    y_train_t = dataset["y_train_t"]
    y_test_t = dataset["y_test_t"]
    train_int_labels = dataset["train_int_labels"]
    indices_by_class = dataset["indices_by_class"]
    num_classes = dataset["num_classes"]
    class_to_idx = dataset["class_to_idx"]
    print("Loaded dataset from pickle.")
else:
    path = 'ptb-xl-dataset/'
    sampling_rate = 100
    Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    X = load_raw_data(Y, sampling_rate, path)
    agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
    test_fold = 10
    X_train = X[np.where(Y.strat_fold != test_fold)]
    X_test  = X[np.where(Y.strat_fold == test_fold)]
    y_train = np.array(Y[(Y.strat_fold != test_fold)].diagnostic_superclass)
    y_test  = np.array(Y[Y.strat_fold == test_fold].diagnostic_superclass)
    unique_classes = np.unique(np.concatenate(Y['diagnostic_superclass'].values))
    print(unique_classes)
    # ['CD' 'HYP' 'MI' 'NORM' 'STTC']
    #
    num_classes = len(unique_classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    def one_hot_encode_local(labels, num_classes):
        v = np.zeros(num_classes, dtype=np.float32)
        for lbl in labels:
            v[class_to_idx[lbl]] = 1.0
        return v
    y_train_ohe = np.array([one_hot_encode_local(lbls, num_classes) for lbls in y_train])
    y_test_ohe  = np.array([one_hot_encode_local(lbls, num_classes) for lbls in y_test])
    y_train_t = torch.tensor(y_train_ohe, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test_ohe, dtype=torch.float32, device=device)
    train_int_labels = torch.argmax(y_train_t, dim=1)
    indices_by_class = {c: (train_int_labels == c).nonzero(as_tuple=True)[0] for c in range(num_classes)}
    dataset = {"X_train_t": X_train_t, "X_test_t": X_test_t, "y_train_t": y_train_t, "y_test_t": y_test_t,
               "train_int_labels": train_int_labels, "indices_by_class": indices_by_class,
               "num_classes": num_classes, "class_to_idx": class_to_idx}
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)
    print("Processed dataset and saved to pickle.")

umap_path = "umap_results.pkl"
if os.path.exists(umap_path):
    with open(umap_path, "rb") as f:
        distance_matrix = pickle.load(f)
    print("Loaded UMAP results from pickle.")
else:
    X_all = torch.cat([X_train_t, X_test_t], dim=0)
    X_all_flat = X_all[:, :, 0].cpu().numpy()
    reducer = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.99,
                        spread=2.0, set_op_mix_ratio=1.0, local_connectivity=1,
                        repulsion_strength=15.0, negative_sample_rate=10,
                        n_epochs=10000, learning_rate=1, init='spectral',
                        random_state=42, metric='euclidean', verbose=True)
    embedding = reducer.fit_transform(X_all_flat)
    y_train_int = torch.argmax(y_train_t, dim=1).cpu().numpy()
    y_test_int = torch.argmax(y_test_t, dim=1).cpu().numpy()
    y_all = np.concatenate([y_train_int, y_test_int], axis=0)
    centroids = {}
    umap_class_cloud_idx = {}
    umap_class_cloud_emb = {}
    for c in range(num_classes):
        idx = np.where(y_all == c)[0]
        umap_class_cloud_idx[str(c)] = idx
        umap_class_cloud_emb[str(c)] = embedding[idx, :]
    break_label = False
    emb = umap_class_cloud_emb.copy()
    distance_matrix = np.zeros((X_all_flat.shape[0], num_classes, 3))
    for i in range(X_all_flat.shape[0]):
        if break_label:
            break
        for c in range(num_classes):
            idx = umap_class_cloud_idx[str(c)]
            if umap_class_cloud_emb[str(c)].shape[0] == 0:
                if i == X_all_flat.shape[0] and c == num_classes:
                    break_label = True
                continue
            if idx[0] == i:
                pos_val = umap_class_cloud_emb[str(c)][0]
                distance_matrix[i, c, 0:2] = pos_val
                distance_matrix[i, c, 2] = 1.
                umap_class_cloud_idx[str(c)] = umap_class_cloud_idx[str(c)][1:]
                umap_class_cloud_emb[str(c)] = umap_class_cloud_emb[str(c)][1:]
                break
            else:
                pass
        if not break_label:
            idx = distance_matrix[i, :, 2] == 1.
            idx_other = [k for k, x in enumerate(idx) if not x]
            idx_center = [k for k, x in enumerate(idx) if x]
            other_vals_x = [(np.abs(emb[str(k)][:,0]-distance_matrix[i, idx_center, 0])).argmin() for k in idx_other]
            other_vals_y = [(np.abs(emb[str(k)][:,1]-distance_matrix[i, idx_center, 1])).argmin() for k in idx_other]
            l = 0
            for indx in idx_other:
                distance_matrix[i, indx, 0] = emb[str(indx)][other_vals_x[l],0]
                distance_matrix[i, indx, 1] = emb[str(indx)][other_vals_y[l],1]
                l += 1
        else:
            print(umap_class_cloud_idx)
            print(i, X_all_flat.shape[0])
    with open(umap_path, "wb") as f:
        pickle.dump(distance_matrix, f)
    print("Computed UMAP results and saved to pickle.")

def calculate_relative_positions(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 3 or matrix.shape[1] != 5 or matrix.shape[2] != 3:
        raise ValueError("Input matrix must be of shape (B,5,3)")
    relative_matrix = matrix.copy()
    ref_indices = np.argmax(relative_matrix[:, :, 2], axis=1)
    B = relative_matrix.shape[0]
    ref_coords = relative_matrix[np.arange(B), ref_indices, :2]
    relative_matrix[:, :, :2] = relative_matrix[:, :, :2] - ref_coords[:, None, :]
    return relative_matrix

num_epochs = 10000
batch_size = 32
lead_view = 0
seq_len = 1000
model = ECGClassifier(num_classes, seq_len).to(device)
print('Model Parameters: ', count_parameters(model))
class_weights = torch.ones(num_classes, device=device)
class_weights_2 = torch.ones(2, device=device)

criterion_main = nn.CrossEntropyLoss()#FocalLoss(gamma=2.0, class_weight=class_weights)
criterion_aux = nn.CrossEntropyLoss()#FocalLoss(gamma=2.0, class_weight=class_weights_2)#
criterion_signals = nn.MSELoss()
#
optimizer = optim.Adam(list(model.parameters()), lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=5e-6, amsgrad=True)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []
umap_coeff_train = distance_matrix[0:len(X_train_t)]
umap_coeff_test = distance_matrix[len(X_train_t):len(X_train_t)+len(X_test_t)]
umap_coeff_train = np.abs(calculate_relative_positions(umap_coeff_train))*1e2
umap_coeff_test = np.abs(calculate_relative_positions(umap_coeff_test))*1e2
D_tensor = torch.tensor(umap_coeff_train, device=device, dtype=torch.float32)
D_tensor_test = torch.tensor(umap_coeff_test, device=device, dtype=torch.float32)
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
    train_data_signal_sample = X_train_t[random_train_indices, :]

    train_data_sample = augment_data(train_data_sample)
    test_data_sample = X_test_t[random_test_indices, :, lead_view]
    test_label_sample = y_test_t[random_test_indices]
    test_data_signal_sample = X_test_t[random_test_indices, :]

    model.train()
    main_logits, aux_logits, features = model(train_data_sample)
    features = augment_data(features)
    main_targets = torch.argmax(train_label_sample, dim=1)
    aux_targets = torch.nn.functional.one_hot((main_targets == 2).long(), num_classes=2).float()
    fused_train_logits = fuse_logits_dynamic(main_logits, aux_logits[:, 1])
    main_loss = criterion_main(fused_train_logits, main_targets)

    aux_loss = criterion_aux(aux_logits, torch.argmax(aux_targets, dim=1))
    signal_loss = criterion_signals(features, train_data_signal_sample)
    preds_train = torch.argmax(fused_train_logits, dim=1)
    batch_D_tensor = D_tensor[random_train_indices]
    penalty_vals = torch.mean(batch_D_tensor[torch.arange(batch_D_tensor.size(0)), preds_train, :2], dim=1)
    mask = (preds_train != main_targets).float()
    umap_loss = torch.mean(penalty_vals * mask)
    total_loss = (main_loss +aux_loss +signal_loss)*(1+umap_loss)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    train_prob = fused_train_logits.softmax(dim=1)
    _, train_preds = torch.max(train_prob, dim=1)
    train_correct = (train_preds == main_targets).sum().item()
    train_acc = train_correct / main_targets.size(0)
    model.eval()
    with torch.no_grad():
        main_test_logits, aux_test_logits, test_features = model(test_data_sample)
        test_targets = torch.argmax(test_label_sample, dim=1)
        aux_test_targets = torch.nn.functional.one_hot((test_targets == 2).long(), num_classes=2).float()
        fused_test_logits = fuse_logits_dynamic(main_test_logits, aux_test_logits[:, 1])
        test_main_loss = criterion_main(fused_test_logits, test_targets)
        test_aux_loss = criterion_aux(aux_test_logits, torch.argmax(aux_test_targets, dim=1))
        preds_test = torch.argmax(fused_test_logits, dim=1)
        batch_test_D_tensor = D_tensor_test[random_test_indices]
        test_penalty_vals = torch.mean(batch_test_D_tensor[torch.arange(test_data_sample.size(0)), preds_test, :2], dim=1)
        mask_test = (preds_test != test_targets).float()
        umap_test_loss = torch.mean(test_penalty_vals * mask_test)
        test_signal_loss = criterion_signals(test_features, test_data_signal_sample)
        test_loss = (test_main_loss + test_aux_loss +test_signal_loss)*(1+ umap_test_loss)
        test_prob = fused_test_logits.softmax(dim=1)
        _, test_preds = torch.max(test_prob, dim=1)
        test_correct = (test_preds == test_targets).sum().item()
        test_acc = test_correct / test_targets.size(0)
    if epoch % 10 == 0:
        train_loss_history.append(total_loss.item())
        test_loss_history.append(test_loss.item())
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
    if epoch % 50 == 0 and epoch > 0:
        end_block_time = time.time()
        block_elapsed = end_block_time - start_block_time
        mean_block_time = block_elapsed / 50.0
        print(f"[{epoch:04d}/{num_epochs}] Total Loss: {total_loss.item():.4f} | Test Loss: {test_loss.item():.4f} | Train Acc: {train_acc:.2f} | Test Acc: {test_acc:.2f} | Mean Time/Epoch: {mean_block_time:.3f}s")
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
with torch.no_grad():
    random_test_indices = torch.randperm(X_test_t.shape[0], device=device)[:1000]
    test_data_sample = X_test_t[random_test_indices, :, lead_view]
    test_label_sample = y_test_t[random_test_indices]
    main_test_logits, aux_test_logits, _ = model(test_data_sample)
    fused_final_logits = fuse_logits_dynamic(main_test_logits, aux_test_logits[:,1])
    final_prob = fused_final_logits.softmax(dim=1)
    _, preds_full = torch.max(final_prob, dim=1)
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
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm_norm[i, j]:.1f}%", horizontalalignment="center", color="white" if cm_norm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()
