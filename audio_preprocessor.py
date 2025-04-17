import torch
import argparse
import torchaudio
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def preprocess_audio(waveform, sr, target_sr=16000, target_sec=3.0):
    """
    Preprocess an input waveform for deep learning models.

    This function performs the following operations:
    1. Converts multi-channel audio to mono by averaging channels.
    2. Resamples the audio to the target sampling rate (default: 16 kHz).
    3. Normalizes the waveform to the range [-1, 1].
    4. Trims or pads the waveform to a fixed duration (default: 3 seconds).

    Parameters:
    ----------
    waveform : torch.Tensor
        A 2D tensor of shape [channels, samples], typically loaded with torchaudio.load().

    sr : int
        Original sampling rate of the input waveform.

    target_sr : int, default=16000
        Desired sampling rate for output waveform.

    target_sec : float, default=3.0
        Desired fixed duration (in seconds) for output waveform.

    Returns:
    -------
    waveform : torch.Tensor
        A normalized, mono, resampled waveform of shape [1, target_sr * target_sec].
        The waveform will be either zero-padded or truncated to match the fixed duration.

    Example:
    --------
    >>> waveform, sr = torchaudio.load("example.wav")
    >>> processed = preprocess_audio(waveform, sr)
    >>> processed.shape
    torch.Size([1, 48000])
    """
    num_samples = int(target_sr * target_sec)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    waveform = waveform / waveform.abs().max()

    if waveform.shape[1] < num_samples:
        pad = num_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :num_samples]

    return waveform


def preprocess_from_csv(csv_path, input_dir, output_root, label_map=None, use_full_path=False, class_limit=None):
    """
    Preprocess audio files listed in a CSV file and save them as torch tensors (.pt), separated by class.

    This function reads a CSV file containing paths and labels of audio samples, loads each audio file,
    applies waveform-level preprocessing (resampling, mono conversion, normalization, trimming/padding),
    and saves the result to a specified output directory, organizing the data into 'real_processed' and 
    'fake_processed' folders based on the label.

    Parameters:
    ----------
    csv_path : str or Path
        Path to the input CSV file. The first column should contain filenames or paths to audio files,
        and the second column should contain the corresponding labels (e.g., "real", "fake", "bonafide", "spoof").

    input_dir : str or Path
        Directory where the original audio files are located. If `use_full_path=True`, this is ignored.

    output_root : str or Path
        Root directory where the processed tensors will be saved. Two subfolders will be created:
        'real_processed' and 'fake_processed'.

    label_map : dict, optional
        A dictionary mapping label strings in the CSV to binary class labels (0 = real, 1 = fake).
        If None, defaults to using 'real' or 'bona-fide' as 0, everything else as 1.

    use_full_path : bool, default=False
        If True, file paths in the CSV are treated as absolute or full paths and used directly.
        If False, paths are considered relative to `input_dir`.

    class_limit : int, optional
        Maximum number of samples to process per class. If None, all samples are processed.

    Notes:
    -----
    - Assumes audio files are in .flac or .wav format.
    - Skips already processed files (.pt) to avoid redundancy.
    - Uses the preprocess_audio() function defined in this file.
    - Progress is printed and errors are logged without stopping execution.

    Output:
    -------
    Saved .pt tensor files (one per audio clip) in the following structure:
        output_root/
            ├── real_processed/
            └── fake_processed/

    Example:
    --------
    >>> preprocess_from_csv("data/labels.csv", "data/audio", "data/audio_processed")
    """
    input_dir = Path(input_dir)
    output_root = Path(output_root)
    output_real = output_root / "real_processed"
    output_fake = output_root / "fake_processed"
    output_real.mkdir(parents=True, exist_ok=True)
    output_fake.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    real = 0
    fake = 0

    print(f"Preprocessing from CSV: {csv_path}")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if class_limit and real >= class_limit and fake >= class_limit:
            break
        file_key = row[0]
        label_raw = row[1]

        if label_map:
            label = label_map[label_raw.lower()]
        else:
            label = 0 if label_raw.lower() == "bona-fide" or label_raw.lower() == "real" else 1

        if label == 0 and real >= class_limit:
            continue
        if label == 1 and fake >= class_limit:
            continue

        if use_full_path:
            audio_path = Path(file_key)
        else:
            if file_key.endswith(".flac") or file_key.endswith(".wav"):
                audio_path = input_dir / file_key
            else:
                audio_path = input_dir / (file_key + ".flac")

        save_dir = output_real if label == 0 else output_fake
        save_path = save_dir / (audio_path.stem + ".pt")

        if save_path.exists():
            continue

        try:
            waveform, sr = torchaudio.load(audio_path)
            waveform = preprocess_audio(waveform, sr)
            torch.save(waveform, save_path)
            if label == 0:
                real += 1
            else:
                fake += 1
        except Exception as e:
            print(f"Error with {audio_path.name}: {e}")

    print(f"Completed: {len(list(output_real.glob('*.pt')))} real, {len(list(output_fake.glob('*.pt')))} fake saved.")

if __name__ == "__main__":
    # Read parameters from command line
    parser = argparse.ArgumentParser(description="Preprocess audio files from a CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.", required=True)
    parser.add_argument("input_dir", type=str, help="Directory of the original audio files.", required=True)
    parser.add_argument("output_root", type=str, help="Root directory for processed tensors.", required=True)
    parser.add_argument("--label_map", type=str, help="Path to a JSON file for label mapping.", required=False)
    parser.add_argument("--use_full_path", action="store_true", help="Use full paths from CSV.", required=False)
    parser.add_argument("--class_limit", type=int, help="Limit samples per class.", required=False)
    args = parser.parse_args()
    label_map = None
    use_full_path = False
    if args.use_full_path:
        use_full_path = True
    if args.label_map:
        import json
        with open(args.label_map, 'r') as f:
            label_map = json.load(f)
        preprocess_from_csv(args.csv_path, args.input_dir, args.output_root, label_map, args.use_full_path, args.class_limit)
    else:
        preprocess_from_csv(args.csv_path, args.input_dir, args.output_root, None, args.use_full_path, args.class_limit)


