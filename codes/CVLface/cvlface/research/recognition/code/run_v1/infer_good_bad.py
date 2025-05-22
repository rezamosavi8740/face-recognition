import argparse
import os
import shutil
import yaml
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

###############################################################################
# Utility functions
###############################################################################

def parse_yaml(path):
    """Return dictionary loaded from a dataset YAML file."""
    with open(path, 'r') as fh:
        cfg = yaml.safe_load(fh)
    return cfg

###############################################################################
# Dataset
###############################################################################

class ImageFolderDataset(Dataset):
    """A lightweight replacement for torchvision.datasets.ImageFolder that also
    returns the original path so we can copy good / bad samples later."""

    _IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []  # (image_path, class_idx)
        class_to_idx = {}
        for idx, cls_name in enumerate(sorted(d.name for d in self.root.iterdir() if d.is_dir())):
            class_to_idx[cls_name] = idx
        for cls_name, cls_idx in class_to_idx.items():
            cls_dir = self.root / cls_name
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in self._IMG_EXTENSIONS:
                    self.samples.append((img_path, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target, path

###############################################################################
# Model loading
###############################################################################

def load_pretrained(model_path, device):
    """Attempt to load a TorchScript model. If that fails, assume a state_dict.

    Adapt this to your repository's preferred way of loading AdaFace.
    """
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        return model
    except RuntimeError:
        raise RuntimeError(
            "Unable to load the model as a TorchScript file. "
            "If your checkpoint stores only a state_dict, please modify the "
            "`load_pretrained` function so that it instantiates the AdaFace "
            "backbone (e.g., IR_101) and loads the state_dict.")

###############################################################################
# Main inference routine
###############################################################################

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    ds_cfg = parse_yaml(args.yaml)

    # Many CVLface YAMLs store `data_root` as the directory containing class
    # sub