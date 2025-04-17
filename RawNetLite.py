import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual block base
class ResBlock(nn.Module):
    """
    A 1D convolutional residual block for processing sequential data.

    Each block consists of two convolutional layers with BatchNorm and ReLU activation.
    The input is added to the output (residual connection), enabling gradient flow and improving convergence.

    Parameters
    ----------
    channels : int
        Number of input and output channels for the convolutional layers.

    Forward Input
    -------------
    x : Tensor
        Input tensor of shape [B, C, T], where B = batch size, C = channels, T = time steps.

    Forward Output
    --------------
    Tensor
        Output tensor of the same shape [B, C, T].

    Notes
    -----
    - This structure follows the basic residual block idea from ResNet, adapted for 1D audio data.
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)

# RawNetLite model
class RawNetLite(nn.Module):
    """
    RawNetLite: A lightweight end-to-end architecture for audio deepfake detection.

    The model operates directly on raw waveforms and combines convolutional residual blocks 
    for local feature extraction with a bidirectional GRU for temporal modeling. 
    It is optimized for binary classification between real and fake audio.

    Architecture Overview
    ---------------------
    - 1D Conv + BatchNorm + ReLU
    - Three residual blocks (ResBlock)
    - AdaptiveAvgPool to compress the temporal axis
    - Bidirectional GRU to capture long-range dependencies
    - Two fully connected layers ending in sigmoid for probability output

    Input Shape
    -----------
    x : Tensor
        Shape [B, 1, T], where B = batch size, T = number of audio samples (typically 48000 for 3 seconds at 16kHz)

    Output Shape
    ------------
    Tensor
        Shape [B, 1] — probability of the input being a fake (value in [0,1]).

    Example Usage
    -------------
    >>> model = RawNetLite()
    >>> x = torch.randn(16, 1, 48000)  # batch of 3-second audio clips
    >>> y = model(x)
    >>> print(y.shape)  # torch.Size([16, 1])

    Notes
    -----
    - Designed for efficiency and generalization in cross-dataset deepfake detection.
    - Works directly on raw waveform inputs; no handcrafted features or spectrograms needed.
    """
    def __init__(self):
        super(RawNetLite, self).__init__()
        self.conv_pre = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn_pre = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        self.resblock1 = ResBlock(64)
        self.resblock2 = ResBlock(64)
        self.resblock3 = ResBlock(64)

        self.pool = nn.AdaptiveAvgPool1d(64)  # Sequence reduction for GRU

        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=1,
                          batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [B, 1, T]
        x = self.relu(self.bn_pre(self.conv_pre(x)))     # [B, 64, T]
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.pool(x)                                 # [B, 64, 64]

        x = x.transpose(1, 2)                            # [B, 64, 64] → [B, seq, feat]
        output, _ = self.gru(x)                          # [B, 64, 256]
        x = output[:, -1, :]                             # Last step → [B, 256]

        x = self.fc1(x)                                  # [B, 64]
        x = self.fc2(x)                                  # [B, 1]
        return torch.sigmoid(x)
