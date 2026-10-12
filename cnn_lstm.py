import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from imblearn.over_sampling import SMOTE
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device : " + str(device))

train_df = pd.read_csv('train.csv')
val_df   = pd.read_csv('validation.csv')
test_df  = pd.read_csv('test.csv')
train_df = pd.read_csv('train.csv')
val_df   = pd.read_csv('validation.csv')
test_df  = pd.read_csv('test.csv')

for df in [train_df, val_df, test_df]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

X_train = train_df.drop(columns=['label']).values
X_val   = val_df.drop(columns=['label']).values
X_test  = test_df.drop(columns=['label']).values

y_train_raw = train_df['label'].values
y_val_raw   = val_df['label'].valuess
y_test_raw  = test_df['label'].values

print(np.unique(y_train_raw)[:20])

y_train_b = np.where(y_train_raw == 'BenignTraffic', 0, 1)
y_val_b   = np.where(y_val_raw == 'BenignTraffic', 0, 1)
y_test_b  = np.where(y_test_raw == 'BenignTraffic', 0, 1)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train_b)
print("SMOTE : " + str(Counter(y_train_bal)))

def create_sequences(X, y, window_size=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i : i + window_size])
        y_seq.append(y[i + window_size - 1])
    return np.array(X_seq), np.array(y_seq)

print("creation sequences...")
X_train_seq, y_train_seq = create_sequences(X_train_bal, y_train_bal)
X_val_seq,   y_val_seq   = create_sequences(X_val, y_val_b)
X_test_seq,  y_test_seq  = create_sequences(X_test, y_test_b)
print("train : " + str(X_train_seq.shape))
print("val   : " + str(X_val_seq.shape))
print("test  : " + str(X_test_seq.shape))

def make_loader(X, y, batch_size=256, shuffle=True):
    return DataLoader(TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)), batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train_seq, y_train_seq, shuffle=True)
val_loader   = make_loader(X_val_seq,   y_val_seq,   shuffle=False)
test_loader  = make_loader(X_test_seq,  y_test_seq,  shuffle=False)
print("loaders ok")

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=1):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3)
        self.bn1   = nn.BatchNorm1d(64)
        self.relu  = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.lstm    = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

input_dim = X_train_seq.shape[2]
model     = CNNLSTM(input_dim=input_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("modele ok sur " + str(device))

best_val_loss  = float('inf')
patience_count = 0
patience       = 5
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
start = time.time()

for epoch in range(20):
    model.train()
    t_loss, correct, total = 0, 0, 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out  = model(X_b).squeeze()
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        t_loss  += loss.item()
        preds    = (torch.sigmoid(out) > 0.5).float()
        correct += (preds == y_b).sum().item()
        total   += y_b.size(0)

    t_loss /= len(train_loader)
    t_acc   = correct / total * 100
    train_losses.append(t_loss)
    train_accs.append(t_acc)

    model.eval()
    v_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            out  = model(X_b).squeeze()
            loss = criterion(out, y_b)
            v_loss  += loss.item()
            preds    = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == y_b).sum().item()
            total   += y_b.size(0)

    v_loss /= len(val_loader)
    v_acc   = correct / total * 100
    val_losses.append(v_loss)
    val_accs.append(v_acc)

    print("epoch " + str(epoch+1) + " | train loss:" + str(round(t_loss,4)) + " acc:" + str(round(t_acc,2)) + "% | val loss:" + str(round(v_loss,4)) + " acc:" + str(round(v_acc,2)) + "%")

    if v_loss < best_val_loss:
        best_val_loss  = v_loss
        patience_count = 0
        torch.save(model.state_dict(), 'best_cnnlstm.pth')
        print("   meilleur modele sauvegarde")
    else:
        patience_count += 1
        if patience_count >= patience:
            print("early stopping epoch " + str(epoch+1))
            break

duree = time.time() - start
print("termine en " + str(round(duree/60, 1)) + " minutes")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train_losses, color='steelblue', marker='o', label='Train Loss')
ax1.plot(val_losses,   color='crimson',   marker='s', label='Val Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_title('Evolution de la fonction de perte')
ax1.legend()
ax1.grid(True)
ax2.plot(train_accs, color='steelblue', marker='o', label='Train Accuracy')
ax2.plot(val_accs,   color='crimson',   marker='s', label='Val Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Evolution de la precision')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.savefig('courbes_apprentissage.png', dpi=150)
plt.show()

model.load_state_dict(torch.load('best_cnnlstm.pth'))
model.eval()
all_preds, all_labels, all_scores = [], [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        X_b    = X_b.to(device)
        out    = model(X_b).squeeze()
        scores = torch.sigmoid(out)
        preds  = (scores > 0.5).int()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_b.numpy())
        all_scores.extend(scores.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=['Benign', 'Attack'], digits=4))

cm   = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Attack'])
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(cmap='Blues', colorbar=False, ax=ax)
ax.set_title('Matrice de confusion - Classification Binaire')
plt.tight_layout()
plt.savefig('confusion_matrix_binaire.png', dpi=150)
plt.show()

tn, fp, fn, tp = cm.ravel()
print("TP:" + str(tp) + " TN:" + str(tn) + " FP:" + str(fp) + " FN:" + str(fn))

fpr, tpr, _ = roc_curve(all_labels, all_scores)
roc_auc     = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='steelblue', lw=2, label='CNN-LSTM (AUC = ' + str(round(roc_auc, 4)) + ')')
plt.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
plt.plot([0,1], [0,1], color='gray', linestyle='--', label='Aleatoire')
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Courbe ROC - Classification Binaire')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve.png', dpi=150)
plt.show()
print("AUC : " + str(round(roc_auc, 4)))

start_inf = time.time()
with torch.no_grad():
    for X_b, _ in test_loader:
        X_b = X_b.to(device)
        _ = model(X_b)
inf_time = time.time() - start_inf
n = len(test_loader.dataset)
print("temps par flux : " + str(round((inf_time/n)*1000, 4)) + " ms")

from google.colab import files
import os
for f in ['courbes_apprentissage.png', 'confusion_matrix_binaire.png', 'roc_curve.png', 'best_cnnlstm.pth']:
    if os.path.exists(f):
        files.download(f)
        print("telecharge : " + f)