# RawNetLite: Lightweight End-to-End Audio Deepfake Detection

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![PyTorch](https://img.shields.io/badge/framework-pytorch-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)

This repository contains the official implementation of the paper:

> **End-to-end Audio Deepfake Detection from RAW Waveforms: a RawNet-Based Approach with Cross-Dataset Evaluation**  
> *Andrea Di Pierno, Luca Guarnera, Dario Allegra, Sebastiano Battiato*  
> In Proceedings of the **VERIMEDIA Workshop at IJCNN 2025**, Rome, Italy.

---

## 🧠 Overview

**RawNetLite** is a lightweight convolutional-recurrent model designed to detect audio deepfakes directly from raw waveforms, without relying on handcrafted features or large pretrained models.

The model is trained and evaluated under in-domain and cross-dataset scenarios using three public datasets: [FakeOrReal](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset), [AVSpoof2021](https://www.asvspoof.org/index2021.html), and [CodecFake](https://github.com/roger-tseng/CodecFake).

We introduce a training pipeline based on:

- **Raw waveform input**
- **Domain-mix learning**
- **Focal Loss optimization**
- **Waveform-level audio augmentations**

---

## 📄 Paper

If you use this code, please cite our paper:

```bibtex
@inproceedings{dipierno2025rawnetlite,
  title     = {End-to-end Audio Deepfake Detection from RAW Waveforms: a RawNet-Based Approach with Cross-Dataset Evaluation},
  author    = {Andrea Di Pierno and Luca Guarnera and Dario Allegra and Sebastiano Battiato},
  booktitle = {International Joint Conference on Neural Networks (IJCNN) - VERIMEDIA Workshop},
  year      = {2025}
}
```
## 📂 Directory Structure
```bash
.
├── models/                   # Pretrained models
    ├── rawnet_lite.pt                                      # Basic RawNetLite model
    ├── cross_domain_rawnet_lite.pt                         # Cross-domain RawNetLite model
    ├── cross_domain_focal_rawnet_lite.pt                   # Cross-domain RawNetLite with Focal Loss
    ├── triple_cross_domain_focal_rawnet_lite.pt            # Triple cross-domain RawNetLite with Focal Loss
    ├── augmented_triple_cross_domain_focal_rawnet_lite.pt  # Augmented triple cross-domain RawNetLite with Focal Loss
├── .gitignore                # Git ignore file
├── .gitattributes            # Git attributes file
├── audio_preprocessor.py     # Audio preprocessing module
├── AVSpoof_dataset.py        # AVSpoof PyTorch dataset
├── CodecFake_dataset.py      # CodecFake PyTorch dataset
├── focal_loss.py             # Focal loss implementation
├── FOR_dataset.py            # FakeOrReal PyTorch dataset
├── LICENSE                   # License file
├── Mixed_dataset.py          # Mixed-domain PyTorch datasets
├── RawNetLite.py             # Main model architecture
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── tester.py                 # Testing script
└── trainer.py                  # Training script
```
## 🛠 Setup
1. Clone the repository:

```bash
git clone https://github.com/adipiz99/rawnetlite.git
cd RawNetLite
```
2. Install the required packages:

```bash
pip install -r requirements.txt
```
---

## 🔁 Preprocessing

To preprocess your dataset into waveform tensors (.pt):

```bash
python audio_preprocessor.py \
    --csv_path metadata.csv \
    --input_dir path/to/audio \
    --output_root data/audio_processed/
```

This will create `real_processed/` and `fake_processed/` folders with normalized, trimmed, and resampled audio waveforms.

---

## 🧪 Training & evaluation

To run the training script, set the parameters in `trainer.py` and use:

```bash
python trainer.py
```
The script outputs:
- Training loss, validation accuracy and F1 score
- Classification metrics (Precision, Recall, F1) for the validation set
- Equal Error Rate (EER) and threshold for the validation set
- A model trained following the specified parameters

To run the test bench and evaluate all models across all datasets, set the parameters in `tester.py` and use:

```bash
python tester.py
```

The script outputs:
- Classification metrics (Precision, Recall, F1)
- Equal Error Rate (EER) and threshold
- Support for FakeOrReal, AVSpoof2021, CodecFake, and mixed-domain evaluations

Please note that the training and testing scripts need to be run __using different data__, to avoid dataset overlapping.

---

## 🎯 Pretrained Models

Pretrained have been released into the `models/` folder.

- `rawnet_lite.pt`: Basic RawNetLite model trained on the FakeOrReal dataset with BCE Loss.
- `cross_domain_rawnet_lite.pt`: Cross-domain RawNetLite model trained on the FOR dataset and the AVSpoof2021 dataset with BCE Loss.
- `cross_domain_focal_rawnet_lite.pt`: Cross-domain RawNetLite model trained on the FOR dataset and the AVSpoof2021 dataset with Focal Loss.
- `triple_cross_domain_focal_rawnet_lite.pt`: Triple cross-domain RawNetLite model trained on the FOR dataset, the AVSpoof2021 dataset, and the CodecFake dataset with Focal Loss.
- `augmented_triple_cross_domain_focal_rawnet_lite.pt`: Augmented triple cross-domain RawNetLite model trained on the FOR dataset, the AVSpoof2021 dataset, and the CodecFake dataset with Focal Loss and augmentation.

---

## 🗂 Datasets

This repository supports the following datasets:
- [FakeOrReal](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
- [AVSpoof2021](https://www.asvspoof.org/index2021.html)
- [CodecFake](https://github.com/roger-tseng/CodecFake)

Each must be preprocessed using the provided script. Ensure correct splits for training and evaluation.

---

## ⚖ License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 📬 Contact

If you have questions or find this project useful, feel free to contact us:

- Andrea Di Pierno — [andrea.dipierno@imtlucca.it](mailto:andrea.dipierno@imtlucca.it)

---

## 📌 Acknowledgments

This study has been partially supported by SERICS (PE00000014) under the MUR National Recovery and Resilience Plan funded by the European Union - NextGenerationEU
