import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random

class AVSpoofTestDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed audio samples from the AVSpoof2021 dataset.

    This dataset is used during testing and evaluation, loading `.pt` files representing
    3-second normalized waveforms saved via torch.save().

    Parameters
    ----------
    real_dir : str or Path
        Directory containing torch-saved tensors of real (bonafide) audio samples.

    fake_dir : str or Path
        Directory containing torch-saved tensors of fake (spoofed) audio samples.

    max_real : int, optional (default=50000)
        Maximum number of real samples to include in the test set.

    max_fake : int, optional (default=50000)
        Maximum number of fake samples to include in the test set.

    Attributes
    ----------
    samples : list of tuples
        List containing (Path, label) pairs, where label is 0 for real and 1 for fake.

    Notes
    -----
    - All `.pt` files in real_dir and fake_dir are assumed to contain 1x48000 torch tensors.
    - Samples are randomly subsampled (if necessary) to avoid test set imbalance.
    - The list is shuffled after loading for randomized batching.
    - If a file is found to be corrupted or unreadable, it is automatically removed and skipped.

    Example
    -------
    >>> dataset = AVSpoofTestDataset("path/to/real", "path/to/fake", max_real=45000, max_fake=30000)
    >>> waveform, label = dataset[0]
    >>> waveform.shape
    torch.Size([1, 48000])
    >>> label
    0 or 1
    """
    def __init__(self, real_dir, fake_dir, max_real=50000, max_fake=50000):
        self.samples = []

        real_dir = Path(real_dir)
        fake_dir = Path(fake_dir)

        real_files = list(real_dir.glob("*.pt"))
        fake_files = list(fake_dir.glob("*.pt"))
        real_files = random.sample(real_files, min(len(real_files), max_real))
        fake_files = random.sample(fake_files, min(len(fake_files), max_fake))

        self.samples += [(p, 0) for p in real_files]
        self.samples += [(p, 1) for p in fake_files]

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            tensor = torch.load(path)
        except:
            os.remove(path)
            print(f"File {path} corrupted, removed.")
        return tensor, label