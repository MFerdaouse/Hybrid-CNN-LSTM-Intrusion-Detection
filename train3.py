import os
# --- CORRECTION BUG WINDOWS 11 (Loky/WMIC) ---
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import time
import gc
import glob
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# =========================================================
# 1. CONFIGURATION & VÉRIFICATION DU MATÉRIEL
# =========================================================
MODEL_PATH = 'best_cnnlstm.pth'
WINDOW_SIZE = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*60)
print(f"🚀 DÉMARRAGE DU PIPELINE IOT - Matériel : {device}")
print("="*60)

def get_csv_from_folder(folder_name):
    if not os.path.exists(folder_name):
        raise FileNotFoundError(f"ERREUR : Le dossier '{folder_name}' est introuvable.")
    csv_files = glob.glob(os.path.join(folder_name, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"ERREUR : Aucun fichier .csv trouvé dans '{folder_name}'.")
    return csv_files[0]

# =========================================================
# 2. CHARGEMENT INDÉPENDANT DES DONNÉES
# =========================================================
print("\n[ÉTAPE 1] Chargement des données Train, Validation et Test...")

path_train = get_csv_from_folder('train')
path_val   = get_csv_from_folder('validation')
path_test  = get_csv_from_folder('test')

df_train = pd.read_csv(path_train)
df_val   = pd.read_csv(path_val)
df_test  = pd.read_csv(path_test)

for df in [df_train, df_val, df_test]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

cols_constantes = [col for col in df_train.columns if df_train[col].nunique() <= 1 and col != 'label']
df_train.drop(columns=cols_constantes, inplace=True, errors='ignore')
df_val.drop(columns=cols_constantes, inplace=True, errors='ignore')
df_test.drop(columns=cols_constantes, inplace=True, errors='ignore')

X_train = df_train.drop(columns=['label']).values
y_train_raw = df_train['label'].values
X_val = df_val.drop(columns=['label']).values
y_val_raw = df_val['label'].values
X_test = df_test.drop(columns=['label']).values
y_test_raw = df_test['label'].values

del df_train, df_val, df_test
gc.collect()

# =========================================================
# 3. ENCODAGE ET NORMALISATION
# =========================================================
le = LabelEncoder()
le.fit(np.concatenate([y_train_raw, y_val_raw, y_test_raw]))
num_classes = len(le.classes_)

y_train_m = le.transform(y_train_raw)
y_val_m   = le.transform(y_val_raw)
y_test_m  = le.transform(y_test_raw)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# =========================================================
# 4. FENÊTRAGE GLISSANT (CORRIGÉ : AVANT ÉQUILIBRAGE)
# =========================================================
print("\n[ÉTAPE 2] Transformation des paquets en séquences temporelles...")
def create_sequences(X, y, window_size=WINDOW_SIZE):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i : i + window_size])
        y_seq.append(y[i + window_size - 1])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_m)
X_val_seq,   y_val_seq   = create_sequences(X_val_scaled, y_val_m)
X_test_seq,  y_test_seq  = create_sequences(X_test_scaled, y_test_m)

# =========================================================
# 5. ÉQUILIBRAGE HYBRIDE (CORRIGÉ : ADAPTÉ À LA 3D)
# =========================================================
print("\n[ÉTAPE 3] Équilibrage intelligent des classes pour économiser la RAM...")
MAX_SAMPLES = 15000 

# Astuce : On aplatit les séquences 3D en 2D pour que Imblearn puisse les traiter
N, W, F = X_train_seq.shape
X_train_flat = X_train_seq.reshape(N, W * F)

counts = Counter(y_train_seq)

under_strategy = {k: min(v, MAX_SAMPLES) for k, v in counts.items()}
rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)
X_under_flat, y_under = rus.fit_resample(X_train_flat, y_train_seq)

over_strategy = {k: MAX_SAMPLES for k in counts.keys()}
ros = RandomOverSampler(sampling_strategy=over_strategy, random_state=42)
X_train_bal_flat, y_train_bal = ros.fit_resample(X_under_flat, y_under)

# On redimensionne en 3D pour le CNN-LSTM
X_train_bal = X_train_bal_flat.reshape(-1, W, F)
print(f" -> Dataset équilibré à {X_train_bal.shape[0]} séquences.")

# =========================================================
# 6. PYTORCH DATALOADERS
# =========================================================
def make_loader(X, y, batch_size=256, shuffle=True):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=(device.type == 'cuda'))

train_loader = make_loader(X_train_bal, y_train_bal, shuffle=True)
val_loader   = make_loader(X_val_seq,   y_val_seq,   shuffle=False)
test_loader  = make_loader(X_test_seq,  y_test_seq,  shuffle=False)

# =========================================================
# 7. DÉFINITION DU MODÈLE HYBRIDE CNN-LSTM (TON ORIGINAL)
# =========================================================
class RobustCNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=33):
        super(RobustCNNLSTM, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        return self.classifier(x)

input_dim = X_train_seq.shape[2]
model = RobustCNNLSTM(input_dim=input_dim, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# =========================================================
# 8. ENTRAÎNEMENT DU MODÈLE (BOUCLE PRINCIPALE)
# =========================================================
print(f"\n[ÉTAPE 4] Lancement de l'entraînement sur {num_classes} classes...")
best_val_loss = float('inf')
patience_count, patience = 0, 5

for epoch in range(20):
    model.train()
    t_loss, correct, total = 0, 0, 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        t_loss += loss.item()
        preds = torch.argmax(out, dim=1)
        correct += (preds == y_b).sum().item()
        total += y_b.size(0)

    t_loss /= len(train_loader)
    t_acc = correct / total * 100

    model.eval()
    v_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            out = model(X_b)
            loss = criterion(out, y_b)
            v_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)

    v_loss /= len(val_loader)
    v_acc = correct / total * 100

    print(f"Epoch {epoch+1:02d} | Train Loss: {t_loss:.4f} Acc: {t_acc:.2f}% | Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}%")

    if v_loss < best_val_loss:
        best_val_loss = v_loss
        patience_count = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        patience_count += 1
        if patience_count >= patience:
            print(f"-> Arrêt prématuré (Early Stopping) à l'epoch {epoch+1}.")
            break

# =========================================================
# 9. ÉVALUATION FINALE SUR LE TEST SET
# =========================================================
print("\n[ÉTAPE 5] Évaluation finale sur les données de Test...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        X_b = X_b.to(device)
        out = model(X_b)
        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_b.numpy())

class_names = le.inverse_transform(np.arange(num_classes))
print("\n" + "="*60)
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(14, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', colorbar=True, ax=ax, xticks_rotation='vertical')
ax.set_title('Matrice de Confusion Finale')
plt.tight_layout()
plt.savefig('confusion_matrix_final.png', dpi=150)
print("-> Graphique 'confusion_matrix_final.png' sauvegardé.")

# =========================================================
# 10. BONUS PFE : SIMULATION EN TEMPS RÉEL
# =========================================================
print("\n" + "="*60)
print("🛡️ DÉMARRAGE DU SIMULATEUR DE SURVEILLANCE TEMPS RÉEL 🛡️")
print("="*60)
time.sleep(2)

# Sélection de 30 paquets aléatoires dans l'ensemble de test
num_simulations = 30
indices = np.random.choice(len(X_test_seq), num_simulations, replace=False)
alert_count = 0

print(f"{'ID FLUX':<10} | {'STATUT':<12} | {'TYPE D\'ACTIVITÉ':<25} | {'CONFIANCE':<10} | {'LATENCE'}")
print("-" * 80)

with torch.no_grad():
    for i, idx in enumerate(indices):
        tensor_window = torch.FloatTensor(X_test_seq[idx]).unsqueeze(0).to(device)
        
        t0 = time.perf_counter()
        output = model(tensor_window)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        confidence, predicted_class_idx = torch.max(probabilities, dim=0)
        t1 = time.perf_counter()
        
        latency_ms = (t1 - t0) * 1000
        predicted_label = le.inverse_transform([predicted_class_idx.cpu().numpy()])[0]
        conf_pct = confidence.item() * 100
        
        flux_id = f"#{1000 + i}"
        
        if predicted_label == "BenignTraffic":
            status = "🟢 SECURE"
        else:
            alert_count += 1
            status = "🔴 ALERTE"
            
        print(f"{flux_id:<10} | {status:<12} | {predicted_label:<25} | {conf_pct:>6.2f}%   | {latency_ms:>6.2f} ms")
        time.sleep(0.15)

print("\n" + "="*60)
print(f"📊 SURVEILLANCE TERMINÉE : {alert_count} menaces bloquées sur {num_simulations} paquets analysés.")
print("="*60)