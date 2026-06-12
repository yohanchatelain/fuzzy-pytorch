"""Shared components for the MNIST variable-precision SR experiment.

Contains the CNN architecture and common I/O utilities used across
training, inference, and plotting scripts.
"""

import glob
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """Simple CNN for MNIST classification.

    Architecture:
        Conv2d(1→32, 3×3) → ReLU → Conv2d(32→64, 3×3) → ReLU →
        MaxPool2d(2) → Dropout(0.25) → Flatten →
        Linear(9216→128) → ReLU → Dropout(0.5) → Linear(128→10) →
        LogSoftmax
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def load_runs(precision_dir):
    """Load all activation pickle files for one precision level.

    Args:
        precision_dir: Path to a ``precision_<N>`` directory containing
            ``activations_run_*.pkl`` files.

    Returns:
        List of dicts, one per run, each containing layer activations
        and metadata (target labels, predictions, loss).
    """
    files = sorted(glob.glob(os.path.join(precision_dir, "activations_run_*.pkl")))
    runs = []
    for f in files:
        with open(f, "rb") as fh:
            runs.append(pickle.load(fh))
    return runs
