
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from PIL import Image
from sklearn.model_selection import train_test_split
import pywt
import matplotlib.pyplot as plt
import urllib.request
import wfdb
import csv
import pandas as pd
from tqdm import tqdm



# Paths
RAW_DATA_PATH = 'data/raw'
QRS_PATH = 'data/processed/qrs_segments'
NOTQRS_PATH = 'data/processed/notqrs_segments'
WAVELET_IMG_QRS = 'images/trainingDataset/QRS'
WAVELET_IMG_NOTQRS = 'images/trainingDataset/notQRS'
TEST_IMG_QRS = 'images/testDataset/QRS'
TEST_IMG_NOTQRS = 'images/testDataset/notQRS'
PROCESSED_DATA_PATH = 'data/processed'
CSV_LOG_FILE = 'results/hyperparameter_results.xlsx'
SAVED_MODEL_DIR = 'models'
COMPUTE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def fetch_mitbih_dataset(records, extensions, url):
    ensure_dirs(RAW_DATA_PATH)
    for record in records:
        for ext in extensions:
            fname = f"{record}{ext}"
            fpath = os.path.join(RAW_DATA_PATH, fname)
            if not os.path.exists(fpath):
                urllib.request.urlretrieve(url + fname, fpath)

def segment_qrs_and_background(records, sampling_rate=360, window_ms=150):
    ensure_dirs(QRS_PATH, NOTQRS_PATH)
    window_size = int((window_ms / 1000) * sampling_rate)
    for record in records:
        ecg = wfdb.rdrecord(os.path.join(RAW_DATA_PATH, record))
        ann = wfdb.rdann(os.path.join(RAW_DATA_PATH, record), 'atr')
        signal = ecg.p_signal[:, 0]
        for i, r in enumerate(ann.sample):
            s, e = r - window_size // 2, r + window_size // 2
            if s >= 0 and e < len(signal):
                np.save(os.path.join(QRS_PATH, f"{record}_qrs_{i}.npy"), signal[s:e])
        bg_step = sampling_rate * 2
        count = 0
        for t in range(0, len(signal) - window_size, bg_step):
            if all(abs(t - r) > window_size for r in ann.sample):
                np.save(os.path.join(NOTQRS_PATH, f"{record}_notqrs_{count}.npy"), signal[t:t + window_size])
                count += 1

def generate_wavelet_images(qrs_dir, notqrs_dir, out_qrs, out_notqrs, wavelet='db6', level=4):
    ensure_dirs(out_qrs, out_notqrs)
    def convert(segment, path):
        coeffs = pywt.wavedec(segment, wavelet, level=level)
        cA = coeffs[0]
        cA = (cA - np.min(cA)) / (np.max(cA) - np.min(cA) + 1e-8) * 255
        Image.fromarray(cA.astype(np.uint8)).resize((64, 64)).save(path)
    for fname in os.listdir(qrs_dir):
        segment = np.load(os.path.join(qrs_dir, fname))
        convert(segment, os.path.join(out_qrs, f"{wavelet}_{fname.replace('.npy', '.png')}"))
    for fname in os.listdir(notqrs_dir):
        segment = np.load(os.path.join(notqrs_dir, fname))
        convert(segment, os.path.join(out_notqrs, f"{wavelet}_{fname.replace('.npy', '.png')}"))

def generate_test_dataset_images(wavelet='db6', level=4):
    ensure_dirs(TEST_IMG_QRS, TEST_IMG_NOTQRS)
    def save_plot(segment, path):
        coeffs = pywt.wavedec(segment, wavelet, level=level)
        cA = coeffs[0]
        cA = (cA - np.min(cA)) / (np.max(cA) - np.min(cA) + 1e-8) * 255
        img = Image.fromarray(cA.astype(np.uint8)).resize((64, 64))
        img.convert('RGB').save(path, format='JPEG')

    for idx, fname in enumerate(os.listdir(QRS_PATH)[:20]):
        segment = np.load(os.path.join(QRS_PATH, fname))
        save_plot(segment, os.path.join(TEST_IMG_QRS, f"{wavelet}_QRS_{idx}.jpg"))

    for idx, fname in enumerate(os.listdir(NOTQRS_PATH)[:20]):
        segment = np.load(os.path.join(NOTQRS_PATH, fname))
        save_plot(segment, os.path.join(TEST_IMG_NOTQRS, f"{wavelet}_notQRS_{idx}.jpg"))

    print(f"Test dataset images saved to {TEST_IMG_QRS} and {TEST_IMG_NOTQRS}")

class ECGDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

def prepare_numpy_dataset(img_qrs_path, img_notqrs_path, target_shape=(64, 64)):
    X, y = [], []
    for label, folder in [(1, img_qrs_path), (0, img_notqrs_path)]:
        for img_file in os.listdir(folder):
            img = Image.open(os.path.join(folder, img_file)).convert('L').resize(target_shape)
            X.append(np.array(img) / 255.0)
            y.append(label)
    X = np.expand_dims(np.array(X), axis=1)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    ensure_dirs(PROCESSED_DATA_PATH)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'y_test.npy'), y_test)

class ECGCNN(nn.Module):
    def __init__(self, filters, kernel_size, pool_size, dense_units, batch_size, epochs, optimizer): 
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, filters, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool2d(pool_size)
        self.conv2 = nn.Conv2d(filters, filters * 2, kernel_size=kernel_size, padding=1)
        dummy = torch.zeros(1, 1, 64, 64)
        out = self.pool(torch.relu(self.conv1(dummy)))
        out = self.pool(torch.relu(self.conv2(out)))
        flatten_size = out.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flatten_size, dense_units)
        self.fc2 = nn.Linear(dense_units, 2)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def train_model_and_save_best(configs, wavelet='db6'):
    train_dataset = ECGDataset('data/processed/X_train.npy', 'data/processed/y_train.npy')
    test_dataset = ECGDataset('data/processed/X_test.npy', 'data/processed/y_test.npy')
    
    best_acc = 0
    best_model = None
    best_conf = None
    results = []

    ensure_dirs('results', 'models')

    for i, config in enumerate(configs):
        print(f"\nTraining with config {i+1}: {config}")
        model = ECGCNN(**config).to(COMPUTE_DEVICE)

        optimizer_cls = optim.Adam if config['optimizer'] == 'adam' else optim.SGD
        optimizer = optimizer_cls(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

        for epoch in range(config['epochs']):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(COMPUTE_DEVICE), labels.to(COMPUTE_DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(COMPUTE_DEVICE), labels.to(COMPUTE_DEVICE)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        results.append({**config, 'wavelet': wavelet, 'accuracy': acc})

        print(f"Config {i+1} Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_conf = config

    # Save only the best model
    if best_model:
        model_path = os.path.join(SAVED_MODEL_DIR, f'best_model_{wavelet}.pth')
        torch.save(best_model.state_dict(), model_path)

        # Load existing log if it exists
        if os.path.exists(CSV_LOG_FILE):
            existing_df = pd.read_excel(CSV_LOG_FILE)
            full_df = pd.concat([existing_df, pd.DataFrame(results)], ignore_index=True)
        else:
            full_df = pd.DataFrame(results)

        full_df.to_excel(CSV_LOG_FILE, index=False)

        print(f"\nBest model saved to {model_path}")
        print(f"Best Config: {best_conf}, Accuracy: {best_acc:.4f}")


def evaluate_model_on_test_images(model_path, config, test_qrs_path, test_notqrs_path, wavelet):
    model = ECGCNN(**config).to(COMPUTE_DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X_test, y_test = [], []
    for label, folder in [(1, test_qrs_path), (0, test_notqrs_path)]:
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            img = Image.open(img_path).convert('L').resize((64, 64))
            X_test.append(np.array(img) / 255.0)
            y_test.append(label)

    X_test = np.expand_dims(np.array(X_test), axis=1)
    y_test = np.array(y_test)
    inputs = torch.tensor(X_test, dtype=torch.float32).to(COMPUTE_DEVICE)
    labels = torch.tensor(y_test, dtype=torch.long).to(COMPUTE_DEVICE)

    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

    acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    prec = precision_score(labels.cpu().numpy(), preds.cpu().numpy())
    rec = recall_score(labels.cpu().numpy(), preds.cpu().numpy())
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy())
    auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
    cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())

    print(f"Evaluation on {test_qrs_path} & {test_notqrs_path} with {wavelet}:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}, ROC-AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

def main():
    MITBIH_RECORDS = ['100', '101', '102']
    DATA_EXTENSIONS = ['.atr', '.dat', '.hea']
    MITBIH_URL = "https://physionet.org/files/mitdb/1.0.0/"

    wavelet_configs = [
        {'wavelet': 'db4', 'level': 4},
        {'wavelet': 'db6', 'level': 4},
        {'wavelet': 'sym5', 'level': 4},
        {'wavelet': 'coif1', 'level': 3},
    ]

    fetch_mitbih_dataset(MITBIH_RECORDS, DATA_EXTENSIONS, MITBIH_URL)
    segment_qrs_and_background(MITBIH_RECORDS)

    for wave_cfg in wavelet_configs:
        wavelet = wave_cfg['wavelet']
        level = wave_cfg['level']

        print(f"\n--- Processing with Wavelet: {wavelet}, Level: {level} ---")
        generate_wavelet_images(QRS_PATH, NOTQRS_PATH, WAVELET_IMG_QRS, WAVELET_IMG_NOTQRS, wavelet=wavelet, level=level)
        prepare_numpy_dataset(WAVELET_IMG_QRS, WAVELET_IMG_NOTQRS)
        generate_test_dataset_images(wavelet=wavelet, level=level)
        print(f"Test dataset images saved to {TEST_IMG_QRS} and {TEST_IMG_NOTQRS}")
        print(f"Training model with wavelet: {wavelet}")
        
        # Define model configurations

        configs = [
            {"filters": 4, "kernel_size": 2, "pool_size": (2, 2), "dense_units": 2, "batch_size": 32, "epochs": 2, "optimizer": "adam"},
            {"filters": 4, "kernel_size": 2, "pool_size": (2, 2), "dense_units": 2, "batch_size": 32, "epochs": 2, "optimizer": "sgd"},
            {"filters": 4, "kernel_size": 2, "pool_size": (2, 2), "dense_units": 2, "batch_size": 16, "epochs": 4, "optimizer": "adam"},
            {"filters": 4, "kernel_size": 2, "pool_size": (2, 2), "dense_units": 2, "batch_size": 16, "epochs": 4, "optimizer": "sgd"},
        ]
        train_model_and_save_best(configs, wavelet=wavelet)
        print(f"Evaluating best model for wavelet: {wavelet}")


        best_model_path = os.path.join(SAVED_MODEL_DIR, f'best_model_{wavelet}.pth')
        best_config = configs[1]
        evaluate_model_on_test_images(best_model_path, best_config, TEST_IMG_QRS, TEST_IMG_NOTQRS, wavelet=wavelet)

if __name__ == "__main__":
    main()