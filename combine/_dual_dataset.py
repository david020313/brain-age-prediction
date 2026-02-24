"""
Shared dual-channel dataset for combine methods.

Loads paired C1 (gray matter) and C2 (white matter) NIfTI volumes.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import SimpleITK as sitk
import torch


class DualChannelDataset:
    """Dataset that loads paired C1/C2 NIfTI brain scans.

    Expects:
        - ``data_dir`` contains C1 files (e.g. ``c1_xxx_<age>.nii.gz``)
        - A sibling directory with ``c1_`` replaced by ``c2_`` contains C2 files
    """

    def __init__(self, data_dir, transform=None, num_workers=4):
        self.data_dir = data_dir
        self.c2_dir = data_dir.replace("c1_", "c2_")
        self.transform = transform
        self.num_workers = num_workers
        self.scan_paths, self.labels, self.volumes_c1, self.volumes_c2 = self._load()

    def _load(self):
        paths, labels, vols_c1, vols_c2 = [], [], [], []
        tasks = []
        for f in os.listdir(self.data_dir):
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                c1 = os.path.join(self.data_dir, f)
                c2 = os.path.join(self.c2_dir, f.replace("c1", "c2"))
                age = self._parse_age(f)
                if age is not None:
                    tasks.append((c1, c2, age))

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futs = {ex.submit(self._proc, t): t for t in tasks}
            for fut in as_completed(futs):
                try:
                    v1, v2, p, a = fut.result()
                    paths.append(p); labels.append(a)
                    vols_c1.append(v1); vols_c2.append(v2)
                except Exception as e:
                    print(f"Error: {e}")
        return paths, labels, vols_c1, vols_c2

    @staticmethod
    def _parse_age(filename):
        m = re.search(r'_(\d+)\.nii', filename)
        return int(m.group(1)) if m else None

    def _proc(self, task):
        c1, c2, age = task
        v1 = sitk.GetArrayFromImage(sitk.ReadImage(c1))
        v2 = sitk.GetArrayFromImage(sitk.ReadImage(c2))
        if self.transform:
            v1, v2 = self.transform(v1), self.transform(v2)
        return v1, v2, c1, age

    def __len__(self):
        return len(self.scan_paths)

    def __getitem__(self, idx):
        v1 = torch.tensor(self.volumes_c1[idx]).unsqueeze(0).float()
        v2 = torch.tensor(self.volumes_c2[idx]).unsqueeze(0).float()
        combined = torch.cat([v1, v2], dim=0)
        label = torch.tensor(self.labels[idx], dtype=torch.float).unsqueeze(0)
        return combined, label
