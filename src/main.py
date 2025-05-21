
import os
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import wfdb
import urllib.request
import matplotlib.pyplot as plt
import pywt
from PIL import Image
from sklearn.model_selection import train_test_split
import concurrent.futures

# Paths and Constants
RAW_DATA_PATH = 'data/raw'
PROCESSED_SEGMENT_QRS = 'data/processed/qrs_segments'
PROCESSED_SEGMENT_NOTQRS = 'data/processed/notqrs_segments'
WAVELET_IMG_QRS = 'images/trainingDataset/QRS'
WAVELET_IMG_NOTQRS = 'images/trainingDataset/notQRS'
PROCESSED_DATA_PATH = 'data/processed'
CSV_LOG_FILE = 'results/hyperparameter_results.csv'
SAVED_MODEL_DIR = 'models'
COMPUTE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MITBIH_RECORDS = ['100', '101', '102']
DATA_EXTENSIONS = ['.atr', '.dat', '.hea']
MITBIH_URL = "https://physionet.org/files/mitdb/1.0.0/"

def train_with_hyperparameter_search():
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    train_data = ECGDataset(os.path.join(PROCESSED_DATA_PATH, 'X_train.npy'), os.path.join(PROCESSED_DATA_PATH, 'y_train.npy'))
    test_data = ECGDataset(os.path.join(PROCESSED_DATA_PATH, 'X_test.npy'), os.path.join(PROCESSED_DATA_PATH, 'y_test.npy'))

    best_accuracy = 0
    best_model = None
    best_config = None

    configs = [
        {"filters": 4, "kernel_size": 2, "pool_size": (2, 2), "dense_units": 2, "batch_size": 32, "epochs": 2, "optimizer": "adam"},
        {"filters": 4, "kernel_size": 3, "pool_size": (2, 2), "dense_units": 2, "batch_size": 32, "epochs": 2, "optimizer": "sgd"},
        {"filters": 4, "kernel_size": 4, "pool_size": (2, 2), "dense_units": 4, "batch_size": 32, "epochs": 2, "optimizer": "adam"},
        {"filters": 6, "kernel_size": 2, "pool_size": (2, 2), "dense_units": 4, "batch_size": 32, "epochs": 2, "optimizer": "sgd"},
        {"filters": 6, "kernel_size": 3, "pool_size": (2, 2), "dense_units": 6, "batch_size": 32, "epochs": 2, "optimizer": "adam"},
        {"filters": 6, "kernel_size": 4, "pool_size": (2, 2), "dense_units": 6, "batch_size": 32, "epochs": 2, "optimizer": "sgd"},
        {"filters": 8, "kernel_size": 2, "pool_size": (2, 2), "dense_units": 6, "batch_size": 32, "epochs": 2, "optimizer": "adam"},
        {"filters": 8, "kernel_size": 3, "pool_size": (2, 2), "dense_units": 8, "batch_size": 32, "epochs": 2, "optimizer": "sgd"},
        {"filters": 8, "kernel_size": 4, "pool_size": (2, 2), "dense_units": 8, "batch_size": 32, "epochs": 2, "optimizer": "adam"},
        {"filters": 4, "kernel_size": 2, "pool_size": (2, 2), "dense_units": 8, "batch_size": 32, "epochs": 2, "optimizer": "sgd"},
        {"filters": 6, "kernel_size": 3, "pool_size": (2, 2), "dense_units": 8, "batch_size": 32, "epochs": 2, "optimizer": "adam"},
        {"filters": 8, "kernel_size": 2, "pool_size": (2, 2), "dense_units": 8, "batch_size": 32, "epochs": 2, "optimizer": "sgd"},
    ]

    with open(CSV_LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filters", "Kernel Size", "Pool Size", "Dense Units", "Batch Size", "Epochs", "Optimizer", "Test Loss", "Test Accuracy"])

        for config in configs:
            print(f"Training config: {config}")
            model = ECGCNN(config["filters"], config["kernel_size"], config["pool_size"], config["dense_units"]).to(COMPUTE_DEVICE)
            train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
            test_loader = DataLoader(test_data, batch_size=config["batch_size"])
            optimizer = optim.Adam(model.parameters(), lr=0.001) if config["optimizer"] == "adam" else optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            for _ in range(config["epochs"]):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(COMPUTE_DEVICE), labels.to(COMPUTE_DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    optimizer.step()

            model.eval()
            all_labels, all_preds, losses = [], [], []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(COMPUTE_DEVICE), labels.to(COMPUTE_DEVICE)
                    outputs = model(inputs)
                    loss_value = criterion(outputs, labels).item()
                    preds = torch.argmax(outputs, dim=1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    losses.append(loss_value)

            acc = accuracy_score(all_labels, all_preds)
            avg_loss = np.mean(losses)
            writer.writerow([config["filters"], config["kernel_size"], str(config["pool_size"]), config["dense_units"],
                             config["batch_size"], config["epochs"], config["optimizer"], round(avg_loss, 6), round(acc, 6)])

            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_config = config

    if best_model is not None:
        best_model_path = os.path.join(SAVED_MODEL_DIR, 'best_model.pth')
        torch.save(best_model.state_dict(), best_model_path)
        print(f"Best model saved to {best_model_path} with config: {best_config} and accuracy: {best_accuracy}")

    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs('results', exist_ok=True)


def fetch_mitbih_dataset():
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    for record in MITBIH_RECORDS:
        for ext in DATA_EXTENSIONS:
            file_name = f"{record}{ext}"
            file_url = MITBIH_URL + file_name
            save_location = os.path.join(RAW_DATA_PATH, file_name)
            if not os.path.exists(save_location):
                print(f"Downloading {file_name}...")
                urllib.request.urlretrieve(file_url, save_location)

def segment_qrs_and_background():
    os.makedirs(PROCESSED_SEGMENT_QRS, exist_ok=True)
    os.makedirs(PROCESSED_SEGMENT_NOTQRS, exist_ok=True)
    window_size = int((150 / 1000) * 360)

    for record in MITBIH_RECORDS:
        ecg_record = wfdb.rdrecord(os.path.join(RAW_DATA_PATH, record))
        annotations = wfdb.rdann(os.path.join(RAW_DATA_PATH, record), 'atr')
        signal_data = ecg_record.p_signal[:, 0]
        r_locations = annotations.sample

        for idx, r_pos in enumerate(r_locations):
            start, end = r_pos - window_size // 2, r_pos + window_size // 2
            if start >= 0 and end < len(signal_data):
                np.save(os.path.join(PROCESSED_SEGMENT_QRS, f"{record}_qrs_{idx}.npy"), signal_data[start:end])

        signal_length = len(signal_data)
        step_size = 360 * 2
        counter = 0
        for t in range(0, signal_length - window_size, step_size):
            if all(abs(t - r) > window_size for r in r_locations):
                np.save(os.path.join(PROCESSED_SEGMENT_NOTQRS, f"{record}_notqrs_{counter}.npy"), signal_data[t:t + window_size])
                counter += 1


def convert_segments_to_wavelet_imgs(wavelet_type='db6', level=4):
    os.makedirs(PROCESSED_SEGMENT_QRS, exist_ok=True)
    os.makedirs(PROCESSED_SEGMENT_NOTQRS, exist_ok=True)
    os.makedirs(WAVELET_IMG_QRS, exist_ok=True)
    os.makedirs(WAVELET_IMG_NOTQRS, exist_ok=True)

    def process_and_save(segment_path, output_dir):
        try:
            segment = np.load(segment_path)
            coeffs = pywt.wavedec(segment, wavelet_type, level=level)
            cA = coeffs[0]
            # Normalize to 0-255 for image
            cA = (cA - np.min(cA)) / (np.max(cA) - np.min(cA) + 1e-8) * 255
            img = Image.fromarray(cA.astype(np.uint8)).resize((64, 64))
            fname = os.path.splitext(os.path.basename(segment_path))[0]
            img.save(os.path.join(output_dir, f"{wavelet_type}_{fname}.png"))
        except Exception as e:
            print(f"Error processing {segment_path}: {e}")

    qrs_segments = [os.path.join(PROCESSED_SEGMENT_QRS, f) for f in os.listdir(PROCESSED_SEGMENT_QRS) if f.endswith('.npy')]
    notqrs_segments = [os.path.join(PROCESSED_SEGMENT_NOTQRS, f) for f in os.listdir(PROCESSED_SEGMENT_NOTQRS) if f.endswith('.npy')]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda path: process_and_save(path, WAVELET_IMG_QRS), qrs_segments)
        executor.map(lambda path: process_and_save(path, WAVELET_IMG_NOTQRS), notqrs_segments)


    os.makedirs(PROCESSED_SEGMENT_NOTQRS, exist_ok=True)
    os.makedirs(WAVELET_IMG_QRS, exist_ok=True)
    os.makedirs(WAVELET_IMG_NOTQRS, exist_ok=True)

    def save_wavelet_image(segment, output_path):
        coeffs = pywt.wavedec(segment, wavelet_type, level=level)
        cA = coeffs[0]
        plt.figure(figsize=(2, 2))
        plt.plot(cA)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    for fname in os.listdir(PROCESSED_SEGMENT_QRS):
        segment = np.load(os.path.join(PROCESSED_SEGMENT_QRS, fname))
        save_wavelet_image(segment, os.path.join(WAVELET_IMG_QRS, f"{wavelet_type}_{fname.replace('.npy', '.png')}"))

    for fname in os.listdir(PROCESSED_SEGMENT_NOTQRS):
        segment = np.load(os.path.join(PROCESSED_SEGMENT_NOTQRS, fname))
        save_wavelet_image(segment, os.path.join(WAVELET_IMG_NOTQRS, f"{wavelet_type}_{fname.replace('.npy', '.png')}"))


    os.makedirs(WAVELET_IMG_NOTQRS, exist_ok=True)

    def save_wavelet_image(segment, output_path):
        coeffs = pywt.wavedec(segment, 'db6', level=4)
        cA4, cD4, *_ = coeffs
        plt.figure(figsize=(2, 2))
        plt.plot(cA4)
        plt.plot(cD4)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    for fname in os.listdir(PROCESSED_SEGMENT_QRS):
        segment = np.load(os.path.join(PROCESSED_SEGMENT_QRS, fname))
        save_wavelet_image(segment, os.path.join(WAVELET_IMG_QRS, fname.replace('.npy', '.png')))

    for fname in os.listdir(PROCESSED_SEGMENT_NOTQRS):
        segment = np.load(os.path.join(PROCESSED_SEGMENT_NOTQRS, fname))
        save_wavelet_image(segment, os.path.join(WAVELET_IMG_NOTQRS, fname.replace('.npy', '.png')))

def prepare_image_dataset():
    X, y = [], []
    TARGET_SIZE = (64, 64)
    for label, folder in [(1, WAVELET_IMG_QRS), (0, WAVELET_IMG_NOTQRS)]:
        for img_file in os.listdir(folder):
            img = Image.open(os.path.join(folder, img_file)).convert('L').resize(TARGET_SIZE)
            X.append(np.array(img) / 255.0)
            y.append(label)
    X = np.expand_dims(np.array(X), axis=1)
    y = np.array(y)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'y_test.npy'), y_test)

class ECGDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

class ECGCNN(nn.Module):
    def __init__(self, filters, kernel_size, pool_size, dense_units, input_shape=(1, 64, 64)):
        super(ECGCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, filters, kernel_size=kernel_size, padding=1)
        self.pool = nn.MaxPool2d(pool_size)
        self.conv2 = nn.Conv2d(filters, filters * 2, kernel_size=kernel_size, padding=1)
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = self.pool(torch.relu(self.conv1(dummy)))
            x = self.pool(torch.relu(self.conv2(x)))
            self.flatten_size = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(self.flatten_size, dense_units)
        self.fc2 = nn.Linear(dense_units, 2)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def convert_segments_to_wavelet_imgs(wavelet_type='db6', level=4):
    os.makedirs(PROCESSED_SEGMENT_QRS, exist_ok=True)
    os.makedirs(PROCESSED_SEGMENT_NOTQRS, exist_ok=True)
    os.makedirs(WAVELET_IMG_QRS, exist_ok=True)
    os.makedirs(WAVELET_IMG_NOTQRS, exist_ok=True)

    def process_and_save(segment_path, output_dir):
        try:
            segment = np.load(segment_path)
            coeffs = pywt.wavedec(segment, wavelet_type, level=level)
            cA = coeffs[0]
            # Normalize to 0-255 for image
            cA = (cA - np.min(cA)) / (np.max(cA) - np.min(cA) + 1e-8) * 255
            img = Image.fromarray(cA.astype(np.uint8)).resize((64, 64))
            fname = os.path.splitext(os.path.basename(segment_path))[0]
            img.save(os.path.join(output_dir, f"{wavelet_type}_{fname}.png"))
        except Exception as e:
            print(f"Error processing {segment_path}: {e}")

    qrs_segments = [os.path.join(PROCESSED_SEGMENT_QRS, f) for f in os.listdir(PROCESSED_SEGMENT_QRS) if f.endswith('.npy')]
    notqrs_segments = [os.path.join(PROCESSED_SEGMENT_NOTQRS, f) for f in os.listdir(PROCESSED_SEGMENT_NOTQRS) if f.endswith('.npy')]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(lambda path: process_and_save(path, WAVELET_IMG_QRS), qrs_segments)
        executor.map(lambda path: process_and_save(path, WAVELET_IMG_NOTQRS), notqrs_segments)


    os.makedirs(PROCESSED_SEGMENT_NOTQRS, exist_ok=True)
    os.makedirs(WAVELET_IMG_QRS, exist_ok=True)
    os.makedirs(WAVELET_IMG_NOTQRS, exist_ok=True)

    def save_wavelet_image(segment, output_path):
        coeffs = pywt.wavedec(segment, wavelet_type, level=level)
        cA = coeffs[0]
        plt.figure(figsize=(2, 2))
        plt.plot(cA)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    for fname in os.listdir(PROCESSED_SEGMENT_QRS):
        segment = np.load(os.path.join(PROCESSED_SEGMENT_QRS, fname))
        save_wavelet_image(segment, os.path.join(WAVELET_IMG_QRS, f"{wavelet_type}_{fname.replace('.npy', '.png')}"))

    for fname in os.listdir(PROCESSED_SEGMENT_NOTQRS):
        segment = np.load(os.path.join(PROCESSED_SEGMENT_NOTQRS, fname))
        save_wavelet_image(segment, os.path.join(WAVELET_IMG_NOTQRS, f"{wavelet_type}_{fname.replace('.npy', '.png')}"))


    os.makedirs(WAVELET_IMG_NOTQRS, exist_ok=True)

    def save_wavelet_image(segment, output_path):
        coeffs = pywt.wavedec(segment, wavelet_type, level=level)
        cA = coeffs[0]
        plt.figure(figsize=(2, 2))
        plt.plot(cA)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    for fname in os.listdir(PROCESSED_SEGMENT_QRS):
        segment = np.load(os.path.join(PROCESSED_SEGMENT_QRS, fname))
        save_wavelet_image(segment, os.path.join(WAVELET_IMG_QRS, f"{wavelet_type}_{fname.replace('.npy', '.png')}"))

    for fname in os.listdir(PROCESSED_SEGMENT_NOTQRS):
        segment = np.load(os.path.join(PROCESSED_SEGMENT_NOTQRS, fname))
        save_wavelet_image(segment, os.path.join(WAVELET_IMG_NOTQRS, f"{wavelet_type}_{fname.replace('.npy', '.png')}"))

# Run with multiple wavelet configurations
wavelet_configs = [
    {'wavelet_type': 'db4', 'level': 4},
    {'wavelet_type': 'db6', 'level': 4},
    {'wavelet_type': 'sym5', 'level': 4},
    {'wavelet_type': 'coif1', 'level': 3}
]

def is_dataset_ready():
    expected_files = [
        os.path.join(PROCESSED_DATA_PATH, 'X_train.npy'),
        os.path.join(PROCESSED_DATA_PATH, 'y_train.npy'),
        os.path.join(PROCESSED_DATA_PATH, 'X_test.npy'),
        os.path.join(PROCESSED_DATA_PATH, 'y_test.npy'),
    ]
    images_exist = os.path.exists(WAVELET_IMG_QRS) and os.listdir(WAVELET_IMG_QRS)
    test_images_exist = os.path.exists('images/testDataset/QRS') and os.listdir('images/testDataset/QRS') and \
                        os.path.exists('images/testDataset/notQRS') and os.listdir('images/testDataset/notQRS')
    return all(os.path.exists(f) for f in expected_files) and images_exist and test_images_exist

def generate_test_dataset_images(wavelet_type='db6', level=4):
    test_qrs_path = 'images/testDataset/QRS'
    test_notqrs_path = 'images/testDataset/notQRS'
    os.makedirs(test_qrs_path, exist_ok=True)
    os.makedirs(test_notqrs_path, exist_ok=True)

    def save_wavelet_image(segment, output_path):
        coeffs = pywt.wavedec(segment, wavelet_type, level=level)
        cA = coeffs[0]
        plt.figure(figsize=(2, 2))
        plt.plot(cA)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    for idx, fname in enumerate(os.listdir(PROCESSED_SEGMENT_QRS)[:20]):
        segment = np.load(os.path.join(PROCESSED_SEGMENT_QRS, fname))
        save_wavelet_image(segment, os.path.join(test_qrs_path, f"{wavelet_type}_{fname.replace('.npy', '.png')}"))

    for idx, fname in enumerate(os.listdir(PROCESSED_SEGMENT_NOTQRS)[:20]):
        segment = np.load(os.path.join(PROCESSED_SEGMENT_NOTQRS, fname))
        save_wavelet_image(segment, os.path.join(test_notqrs_path, f"{wavelet_type}_{fname.replace('.npy', '.png')}"))

    print(f"Test dataset images generated in {test_qrs_path} and {test_notqrs_path}")

def run_full_training_pipeline():
    if not is_dataset_ready():
        print("Starting data preparation...")
        fetch_mitbih_dataset()
        segment_qrs_and_background()
        for config in wavelet_configs:
            print(f"Generating images with wavelet {config['wavelet_type']} at level {config['level']}")
            convert_segments_to_wavelet_imgs(wavelet_type=config['wavelet_type'], level=config['level'])
            prepare_image_dataset()
        generate_test_dataset_images()  # Ensure test dataset is also generated
    else:
        print("Dataset and test dataset already exist. Skipping data preparation...")

    train_with_hyperparameter_search()
    print(f"Pipeline completed. Results saved to {CSV_LOG_FILE}")

    if not os.path.exists(RAW_DATA_PATH) or len(os.listdir(RAW_DATA_PATH)) == 0:
        print("Starting data preparation...")
        fetch_mitbih_dataset()
        segment_qrs_and_background()
        convert_segments_to_wavelet_imgs()
        prepare_image_dataset()
    else:
        print("Data already exists. Skipping data preparation...")

    train_with_hyperparameter_search()
    print(f"Pipeline completed. Results saved to {CSV_LOG_FILE}")

if __name__ == "__main__":
    run_full_training_pipeline()


    X_test, y_test = [], []
    for label, folder in [(1, 'images/testDataset/QRS'), (0, 'images/testDataset/notQRS')]:
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
    print(f"Test Accuracy on images/testDataset with wavelet {wavelet_type}: {acc:.4f}")


# Evaluate best model if it exists
best_model_path = os.path.join(SAVED_MODEL_DIR, 'best_model.pth')
if os.path.exists(best_model_path):
    evaluate_model_on_test_dataset(best_model_path)

def evaluate_model_on_test_dataset(model_path, filters, kernel_size, pool_size, dense_units, wavelet_type='db6'):
    print(f"Evaluating best model with config: filters={filters}, kernel_size={kernel_size}, pool_size={pool_size}, dense_units={dense_units}")
    model = ECGCNN(filters=filters, kernel_size=kernel_size, pool_size=pool_size, dense_units=dense_units)
    model.load_state_dict(torch.load(model_path))
    model.to(COMPUTE_DEVICE)
    model.eval()

    X_test, y_test = [], []
    for label, folder in [(1, 'images/testDataset/QRS'), (0, 'images/testDataset/notQRS')]:
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            img = Image.open(img_path).convert('L').resize((64, 64))
            X_test.append(np.array(img) / 255.0)
            y_test.append(label)

    if len(X_test) == 0:
        print("Test set is empty.")
        return

    X_test = np.expand_dims(np.array(X_test), axis=1)
    y_test = np.array(y_test)

    inputs = torch.tensor(X_test, dtype=torch.float32).to(COMPUTE_DEVICE)
    labels = torch.tensor(y_test, dtype=torch.long).to(COMPUTE_DEVICE)

    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

    acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    print(f"Test Accuracy on images/testDataset with wavelet {wavelet_type}: {acc:.4f}")


# Evaluate best model if it exists
best_model_path = os.path.join(SAVED_MODEL_DIR, 'best_model.pth')
best_model_config = {'filters': 6, 'kernel_size': 3, 'pool_size': (2, 2), 'dense_units': 6}
if os.path.exists(best_model_path):
    evaluate_model_on_test_dataset(best_model_path, **best_model_config)
