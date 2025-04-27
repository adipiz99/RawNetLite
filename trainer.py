import os
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from focal_loss import FocalLoss
from RawNetLite import RawNetLite
from FOR_dataset import FakeOrRealTestDataset
from AVSpoof_dataset import AVSpoofTestDataset
from CodecFake_dataset import CodecFakeTestDataset
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from Mixed_dataset import DoubleDomainDataset, MultiDomainDataset, AugmentedMultiDomainDataset

# ------------------------------
# PARAMETERS
# ------------------------------
BATCH_SIZE = 16 # Set the batch size for DataLoaders
EPOCHS = 35 # Set the number of epochs for training
LEARNING_RATE = 1e-4 # Set the learning rate for the optimizer
SEED = 42 # Set a random seed for reproducibility
MAX_PER_CLASS = 5000 #Set the maximum number of samples per class for each dataset (cross-domain, Triple-domain, augmented triple-domain)
MAX_REAL = 5000 #Set the maximum number of real samples (single-domain)
MAX_FAKE = 5000 #Set the maximum number of fake samples (single-domain)
LOSS = "focal" # "focal" or "bce

# ------------------------------
# DATASET CONFIGURATION
# ------------------------------
CROSS_DOMAIN = True    #If False, single domain mode is used (ignoring all the parameters above), if True, double domain mode is used (if all the parameters above are set to False)
TRIPLE_DOMAIN = True   #If True and CROSS_DOMAIN is True, triple domain mode is used
AUGMENTATION = True    #If True and CROSS_DOMAIN and TRIPLE_DOMAIN is True, augmented triple domain mode is used

# ------------------------------
# FOLDERS
# ------------------------------
MODEL_ROOT = os.path.join(os.getcwd(), "path/to", "models") # Set the model root directory
MODEL_NAME = "model_name_to_be_used.pt" # Set the model name for saving/loading
DATASET_ROOT_FOR = os.path.join(os.getcwd(), "path/to", "FOR") # Set the dataset root directory for FOR dataset
DATASET_ROOT_AVSPOOF = os.path.join(os.getcwd(), "path/to", "AVSpoof2021") # Set the dataset root directory for AVSpoof dataset
DATASET_ROOT_CODECFAKE = os.path.join(os.getcwd(), "path/to", "CodecFake") # Set the dataset root directory for CodecFake dataset

# ------------------------------
# DATASET LOADING
# ------------------------------
def load_dataset():
    real_dirs = [
        os.path.join(DATASET_ROOT_FOR, "real_processed"),
        os.path.join(DATASET_ROOT_AVSPOOF, "real_processed"),
        os.path.join(DATASET_ROOT_CODECFAKE, "real_processed")
    ]
    fake_dirs = [
        os.path.join(DATASET_ROOT_FOR, "fake_processed"),
        os.path.join(DATASET_ROOT_AVSPOOF, "fake_processed"),
        os.path.join(DATASET_ROOT_CODECFAKE, "fake_processed")
    ]

    if not CROSS_DOMAIN: # Single domain RawNetLite
        dataset = FakeOrRealTestDataset( # FOR dataset (can be replaced with any single-domain dataset)
            real_dir=os.path.join(DATASET_ROOT_FOR, "real_processed"),
            fake_dir=os.path.join(DATASET_ROOT_FOR, "fake_processed"),
            max_real=MAX_REAL,
            max_fake=MAX_FAKE
        )
    elif CROSS_DOMAIN and not TRIPLE_DOMAIN: # Double domain RawNetLite
        real_dirs = [
            os.path.join(DATASET_ROOT_FOR, "real_processed"),
            os.path.join(DATASET_ROOT_AVSPOOF, "real_processed"),
        ]
        fake_dirs = [
            os.path.join(DATASET_ROOT_FOR, "fake_processed"),
            os.path.join(DATASET_ROOT_AVSPOOF, "fake_processed"),
        ]

        dataset = DoubleDomainDataset(
            real_dirs=real_dirs,
            fake_dirs=fake_dirs,
            max_per_class=MAX_PER_CLASS
        )
    elif CROSS_DOMAIN and TRIPLE_DOMAIN and not AUGMENTATION: # Triple domain RawNetLite
        dataset = MultiDomainDataset(
            real_dirs=real_dirs,
            fake_dirs=fake_dirs,
            max_per_class=MAX_PER_CLASS
        )
    elif CROSS_DOMAIN and TRIPLE_DOMAIN and AUGMENTATION: # Augmented triple domain RawNetLite
        dataset = AugmentedMultiDomainDataset(
            real_dirs=real_dirs,
            fake_dirs=fake_dirs,
            max_per_class=MAX_PER_CLASS
        )
    else:
        raise ValueError("Invalid dataset configuration. Please check the parameters.")
    return dataset

# ------------------------------
# TRAINING FUNCTION
# ------------------------------
def train():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    dataset = load_dataset()

    # 80/10/10 split
    # Randomly split the dataset into train, validation, and test sets
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawNetLite().to(device)

    if LOSS == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    elif LOSS == "bce":
        criterion = nn.BCELoss()
    else:
        raise ValueError("Invalid loss function. Choose 'focal' or 'bce'.")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for waveforms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            waveforms = waveforms.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Train Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms = waveforms.to(device)
                labels = labels.to(device)
                outputs = model(waveforms)
                preds = (outputs > 0.5).int().cpu()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.numpy())

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"Validation Accuracy: {acc:.4f} - F1 Score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(MODEL_ROOT, MODEL_NAME)
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model at epoch {epoch+1} with F1 = {f1:.4f}")

    # Test phase
    print("\nEvaluation on test set:")
    model.load_state_dict(torch.load(os.path.join(MODEL_ROOT, MODEL_NAME)))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for waveforms, labels in test_loader:
            waveforms = waveforms.to(device)
            outputs = model(waveforms)
            preds = (outputs > 0.5).int().cpu()
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    print(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    train()
