import torch
import numpy as np
from RawNetLite import RawNetLite
from torch.utils.data import DataLoader
from CodecFake_dataset import CodecFakeTestDataset
from AVSpoof_dataset import AVSpoofTestDataset
from FOR_dataset import FakeOrRealTestDataset
from sklearn.metrics import classification_report, roc_curve
from Mixed_dataset import MultiDomainDataset, AugmentedMultiDomainDataset

# Parameters
MAX_REAL = 5000
MAX_FAKE = 5000
ELEMENTS_PER_CLASS = 25000
BATCH_SIZE = 16
MODEL_ROOT = "/models/"

# Folders
FOR_REAL = "path/to/FOR/real_processed"
FOR_FAKE = "path/to/FOR/fake_processed"
CODECFAKE_REAL = "path/to/CodecFake/real_processed"
CODECFAKE_FAKE = "path/to/CodecFake/fake_processed"
AVSPOOF_REAL = "path/to/AVSpoof2021/real_processed"
AVSPOOF_FAKE = "path/to/AVSpoof2021/fake_processed"

models = [
    "rawnet_lite.pt",
    "cross_domain_rawnet_lite.pt",
    "cross_domain_focal_rawnet_lite.pt",
    "triple_cross_domain_focal_rawnet_lite.pt",
    "augmented_triple_cross_domain_focal_rawnet_lite.pt"
]

# EER
def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, eer_threshold

# Dataset
test_dataset_for = FakeOrRealTestDataset(
    real_dir=FOR_REAL,
    fake_dir=FOR_FAKE,
    max_real=MAX_REAL,
    max_fake=MAX_FAKE
)

test_dataset_codecfake = CodecFakeTestDataset(
    real_dir=CODECFAKE_REAL,
    fake_dir=CODECFAKE_FAKE,
    max_real=MAX_REAL,
    max_fake=MAX_FAKE
)

test_dataset_avspoof = AVSpoofTestDataset(
    real_dir=AVSPOOF_REAL,
    fake_dir=AVSPOOF_FAKE,
    max_real=MAX_REAL,
    max_fake=MAX_FAKE
)

test_dataset_cross = MultiDomainDataset(
    for_real_dir=FOR_REAL,
    for_fake_dir=FOR_FAKE,
    avs_real_dir=AVSPOOF_REAL,
    avs_fake_dir=AVSPOOF_FAKE,
    mix_ratio=0.5,
    max_per_class=ELEMENTS_PER_CLASS
)

test_dataset_triple = AugmentedMultiDomainDataset(
    real_dirs=[FOR_REAL, AVSPOOF_REAL, CODECFAKE_REAL],
    fake_dirs=[FOR_FAKE, AVSPOOF_FAKE, CODECFAKE_FAKE],
    total_per_class=ELEMENTS_PER_CLASS
)

# DataLoader
loaders = [
    DataLoader(test_dataset_for, batch_size=BATCH_SIZE),
    DataLoader(test_dataset_codecfake, batch_size=BATCH_SIZE),
    DataLoader(test_dataset_avspoof, batch_size=BATCH_SIZE),
    DataLoader(test_dataset_cross, batch_size=BATCH_SIZE),
    DataLoader(test_dataset_triple, batch_size=BATCH_SIZE),
]

test_names = [
    "FOR",
    "CodecFake",
    "AVSpoof2021",
    "Cross-dataset (FOR+AVS)",
    "Triple-dataset"
]
print("Beginning test bench...")
for model in models:
    print(f"Loading model {model}...")
    MODEL_PATH = MODEL_ROOT + model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawNetLite().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    for i, test_loader in enumerate(loaders):
        y_true, y_pred, y_scores = [], [], []

        with torch.no_grad():
            for waveforms, labels in test_loader:
                waveforms = waveforms.to(device)
                labels = labels.to(device)
                outputs = model(waveforms).squeeze()

                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(int)

                y_scores.extend(probs)
                y_pred.extend(preds)
                y_true.extend(labels.cpu().numpy())

        if i <= 2:
            print(f"Test {test_names[i]}:")
        else:
            print(f"Test {test_names[i]} (balanced at {ELEMENTS_PER_CLASS} elements per class):")

        print(classification_report(y_true, y_pred, digits=4))
        eer, threshold = compute_eer(y_true, y_scores)
        print(f"Equal Error Rate (EER): {eer:.4f} at threshold {threshold:.4f}\n")
print("Test bench completed!")