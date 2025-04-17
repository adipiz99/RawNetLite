import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch

class DoubleDomainDataset(Dataset):
    """
    A custom PyTorch dataset that combines samples from two domains (e.g., FakeOrReal and AVSpoof2021)
    to simulate cross-domain or domain-mix training scenarios.

    Parameters
    ----------
    for_real_dir : str or Path
        Directory containing preprocessed real samples from the FakeOrReal dataset.
    for_fake_dir : str or Path
        Directory containing preprocessed fake samples from the FakeOrReal dataset.
    avs_real_dir : str or Path
        Directory containing preprocessed real samples from the AVSpoof2021 dataset.
    avs_fake_dir : str or Path
        Directory containing preprocessed fake samples from the AVSpoof2021 dataset.
    mix_ratio : float
        Ratio (0–1) of samples to include from the AVSpoof dataset. The rest will be taken from FakeOrReal.
    max_per_class : int
        Maximum number of real and fake samples to include.
    seed : int
        Seed for random shuffling.

    Output
    ------
    A list of (waveform, label) pairs, shuffled across both domains.
    """
    def __init__(self, 
                 for_real_dir, for_fake_dir,
                 avs_real_dir, avs_fake_dir,
                 mix_ratio=0.2, max_per_class=32000, seed=42):
        self.samples = []

        # Load files
        for_real = list(Path(for_real_dir).glob("*.pt"))
        for_fake = list(Path(for_fake_dir).glob("*.pt"))
        avs_real = list(Path(avs_real_dir).glob("*.pt"))
        avs_fake = list(Path(avs_fake_dir).glob("*.pt"))

        # Limit each source
        for_target = int((1 - mix_ratio) * max_per_class)
        avs_target = int(mix_ratio * max_per_class)

        # Shuffle and select samples
        random.seed(seed)
        selected_real = random.sample(for_real, for_target) + random.sample(avs_real, avs_target)
        selected_fake = random.sample(for_fake, for_target) + random.sample(avs_fake, avs_target)

        self.samples += [(p, 0) for p in selected_real]
        self.samples += [(p, 1) for p in selected_fake]

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = torch.load(path)
        return tensor, label
    
class MultiDomainDataset(Dataset):
    """
    A multi-domain dataset for combining audio samples from multiple sources (e.g., FOR + AVSpoof + CodecFake)
    into a unified training set for triple-domain training.

    Parameters
    ----------
    real_dirs : list of str or Path
        List of directories containing real audio `.pt` files.
    fake_dirs : list of str or Path
        List of directories containing fake audio `.pt` files.
    max_per_class : int
        Number of real and fake samples to use (balanced).
    seed : int
        Seed for reproducibility.

    Output
    ------
    A list of (waveform, label) pairs, shuffled randomly.
    """
    def __init__(self, real_dirs, fake_dirs, max_per_class, seed=42):
        super().__init__()
        self.samples = []

        real_paths = []
        fake_paths = []

        for d in real_dirs:
            real_paths.extend(list(Path(d).glob("*.pt")))
        for d in fake_dirs:
            fake_paths.extend(list(Path(d).glob("*.pt")))

        random.seed(seed)
        random.shuffle(real_paths)
        random.shuffle(fake_paths)

        assert len(real_paths) >= max_per_class, "Not enough real samples"
        assert len(fake_paths) >= max_per_class, "Not enough fake samples"

        real_selected = real_paths[:max_per_class]
        fake_selected = fake_paths[:max_per_class]

        self.samples += [(p, 0) for p in real_selected]
        self.samples += [(p, 1) for p in fake_selected]

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = torch.load(path)
        return tensor, label

class AugmentedMultiDomainDataset(Dataset):
    """
    A multi-domain dataset that includes both original and augmented audio samples 
    for each real and fake file, effectively doubling the dataset size and introducing
    realistic acoustic variability during training.

    Parameters
    ----------
    real_dirs : list of str or Path
        List of directories containing real `.pt` waveforms.
    fake_dirs : list of str or Path
        List of directories containing fake `.pt` waveforms.
    max_per_class : int
        Maximum number of real and fake samples to include (before augmentation).
    seed : int
        Random seed for sample selection and shuffling.

    Augmentations
    -------------
    - Gaussian noise: random amplitude in [0.001, 0.015]
    - Pitch shift: ±2 semitones
    - Time stretch: 0.9x to 1.1x speed
    All transformations are applied with 50% probability using audiomentations.

    Output
    ------
    A dataset of (waveform, label) pairs, including both raw and augmented versions.
    """
    def __init__(self, real_dirs, fake_dirs, max_per_class, seed=42):
        super().__init__()
        self.samples = []
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
        ])

        real_paths = []
        fake_paths = []

        for d in real_dirs:
            real_paths.extend(list(Path(d).glob("*.pt")))
        for d in fake_dirs:
            fake_paths.extend(list(Path(d).glob("*.pt")))

        random.seed(seed)
        random.shuffle(real_paths)
        random.shuffle(fake_paths)

        assert len(real_paths) >= max_per_class, "Not enough real samples"
        assert len(fake_paths) >= max_per_class, "Not enough fake samples"

        real_selected = real_paths[:max_per_class]
        fake_selected = fake_paths[:max_per_class]

        # Duplicate each sample: original + augmented
        for p in real_selected:
            self.samples.append((p, 0, False))  # original
            self.samples.append((p, 0, True))   # augmented

        for p in fake_selected:
            self.samples.append((p, 1, False))  # original
            self.samples.append((p, 1, True))   # augmented

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, apply_aug = self.samples[idx]
        waveform = torch.load(path).float().numpy()

        if apply_aug:
            waveform = self.augment(waveform, sample_rate=16000)

        x = torch.tensor(waveform)
        y = label
        return x, y


