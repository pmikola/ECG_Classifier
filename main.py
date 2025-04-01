# NOTE: Dataset from https://www.kaggle.com/datasets/shayanfazeli/heartbeat/code
#  Publication: https://arxiv.org/pdf/1805.00794
import sys
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from model import EcgClassifier

matplotlib.use('TkAgg')

def plot_correlation_matrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna()
    df = df[[col for col in df if df[col].nunique() > 1]]
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

def plot_scatter_matrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number])
    df = df.dropna()
    df = df[[col for col in df if df[col].nunique() > 1]]
    columnNames = list(df)
    if len(columnNames) > 10:
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

def plot_signal(df,row):
    df = df.dropna()
    plt.plot(df.iloc[row])
    plt.show()


# plot_correlation_matrix(df1, 10)
# plot_scatter_matrix(df1, 10, 10)
# plot_signal(df1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df_train = pd.read_csv('dataset/mitbih_train.csv', delimiter=',', nrows=None)
df_test = pd.read_csv('dataset/mitbih_test.csv', delimiter=',', nrows=None)

df_train_data = df_train.iloc[:, :-1]
df_train_labels = df_train.iloc[:,-1:]

df_test_data = df_test.iloc[:, :-1]
df_test_labels = df_test.iloc[:,-1:]

num_epochs = 20000
batch_size = 64
n_class = 5

model = EcgClassifier(n_class)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []

for epoch in range(0,num_epochs):
    df_test_data = df_test_data.reset_index(drop=True)
    random_train_indices = np.random.choice(df_train.index, size=batch_size, replace=False)
    random_test_indices = np.random.choice(df_test.index, size=batch_size, replace=False)

    df_train_data_sample = df_train_data.loc[random_train_indices]
    df_train_label_sample = df_train_labels.loc[random_train_indices]

    df_test_data_sample = df_test_data.loc[random_test_indices]
    df_test_label_sample = df_test_labels.loc[random_test_indices]

    train_data_tensor = torch.from_numpy(df_train_data_sample.values).float().to(device)
    train_label_tensor = torch.from_numpy(df_train_label_sample.values).long().to(device).squeeze(1)

    test_data_tensor = torch.from_numpy(df_test_data_sample.values).float().to(device)
    test_label_tensor = torch.from_numpy(df_test_label_sample.values).long().to(device).squeeze(1)

    one_hot_train_labels = F.one_hot(train_label_tensor, num_classes=n_class).float()
    one_hot_test_labels = F.one_hot(test_label_tensor, num_classes=n_class).float()

    # NOTE: MODEL TRAIN
    model.train()
    train_outputs = model(train_data_tensor)
    train_loss = criterion(train_outputs, one_hot_train_labels)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    _, train_preds = torch.max(train_outputs, dim=1)
    train_correct = (train_preds == train_label_tensor).sum().item()
    train_acc = train_correct / train_label_tensor.size(0)

    # NOTE: MODEL VALIDATE
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data_tensor)
        test_loss = criterion(test_outputs, one_hot_test_labels)
        _, test_preds = torch.max(test_outputs, dim=1)
        test_correct = (test_preds == test_label_tensor).sum().item()
        test_acc = test_correct / test_label_tensor.size(0)

    if epoch % 10 == 0:
        train_loss_history.append(train_loss.item())
        test_loss_history.append(test_loss.item())
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

    if epoch % 50 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')



plt.style.use('dark_background')

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Loss plot
axs[0].plot( train_loss_history, label='Train Loss')
axs[0].plot( test_loss_history, label='Test Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training and Test Loss')
axs[0].legend()
axs[0].grid(True)

axs[1].plot( train_acc_history, label='Train Accuracy')
axs[1].plot( test_acc_history, label='Test Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Training and Test Accuracy')
axs[1].legend()
axs[1].grid(True)

fig.suptitle("ECG CLASSIFIER RESULTS", fontsize=16)

plt.tight_layout()
plt.show()


model.eval()
with torch.no_grad():
    test_data_tensor_full = torch.from_numpy(df_test_data.values).float().to(device)
    test_label_tensor_full = torch.from_numpy(df_test_labels.values).long().to(device)
    outputs_full = model(test_data_tensor_full)
    _, preds_full = torch.max(outputs_full, dim=1)

cm = confusion_matrix(test_label_tensor_full.cpu().numpy(), preds_full.cpu().numpy())
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(n_class)
plt.xticks(tick_marks, [str(i) for i in range(n_class)])
plt.yticks(tick_marks, [str(i) for i in range(n_class)])
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










