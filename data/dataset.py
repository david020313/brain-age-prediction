"""
Dataset utilities for loading and preprocessing 3D MRI NIfTI volumes.

This module provides:
    - CTScanDataset: PyTorch-compatible dataset that loads NIfTI files from a
      directory and extracts age labels from filenames.
    - Volume I/O and preprocessing functions (read, normalize, resize).
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage


class CTScanDataset:
    """Dataset that loads NIfTI brain scans and extracts age labels from filenames.

    Expected filename format: ``<prefix>_<age>.nii`` or ``<prefix>_<age>.nii.gz``

    Args:
        data_dir: Path to directory containing ``.nii`` / ``.nii.gz`` files.
        transform: Optional callable applied to each volume after loading.
        num_workers: Number of threads for parallel I/O.
    """

    def __init__(self, data_dir, transform=None, num_workers=4):
        self.data_dir = data_dir
        self.transform = transform
        self.num_workers = num_workers
        self.scan_paths, self.labels, self.volumes = self.load_data()

    def load_data(self):
        scan_paths, labels, volumes = [], [], []
        tasks = []

        for filename in os.listdir(self.data_dir):
            if filename.endswith('.nii') or filename.endswith('.nii.gz'):
                file_path = os.path.join(self.data_dir, filename)
                age = self.extract_age_from_filename(filename)
                if age is not None:
                    tasks.append((file_path, age))

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_path = {
                executor.submit(self.process_single_scan, task): task
                for task in tasks
            }
            for future in as_completed(future_to_path):
                task = future_to_path[future]
                try:
                    volume, scan_path, age = future.result()
                    scan_paths.append(scan_path)
                    labels.append(age)
                    volumes.append(volume)
                except Exception as exc:
                    print(f'Scan {task[0]} generated an exception: {exc}')

        return scan_paths, labels, volumes

    @staticmethod
    def extract_age_from_filename(filename):
        """Extract integer age from filename like ``..._<age>.nii(.gz)``."""
        match = re.search(r'_(\d+)\.nii', filename)
        if match:
            return int(match.group(1))
        return None

    def process_single_scan(self, task):
        scan_path, age = task
        volume = process_scan(scan_path)
        if self.transform:
            volume = self.transform(volume)
        return volume, scan_path, age

    def __len__(self):
        return len(self.scan_paths)

    def __getitem__(self, idx):
        volume = self.volumes[idx]
        volume = torch.tensor(volume).unsqueeze(0).float()  # add channel dim
        label = torch.tensor(self.labels[idx], dtype=torch.float).unsqueeze(0)
        return volume, label


# ---------------------------------------------------------------------------
# Volume I/O helpers
# ---------------------------------------------------------------------------

def read_nifti_file(filepath):
    """Read a NIfTI file and return it as a numpy array."""
    scan = sitk.ReadImage(filepath)
    return sitk.GetArrayFromImage(scan)


def normalize(volume):
    """Min-max normalize a volume to [0, 1]."""
    min_val = volume.min()
    max_val = volume.max()
    volume = (volume - min_val) / (max_val - min_val + 1e-8)
    return volume.astype("float32")


def resize_volume(img, desired_shape=(189, 197, 233)):
    """Resize a 3D volume to `desired_shape` via trilinear interpolation.

    Args:
        img: numpy array of shape (depth, height, width).
        desired_shape: Target (depth, height, width). Default matches MNI space.
    """
    factors = [d / c for d, c in zip(desired_shape, img.shape)]
    return ndimage.zoom(img, factors, order=1)


def process_scan(path):
    """Read a NIfTI file and resize to standard dimensions."""
    volume = read_nifti_file(path)
    volume = resize_volume(volume)
    return volume
