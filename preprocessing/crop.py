"""
Background Cropping for Segmented Brain Volumes.

Crops 3D NIfTI volumes to a fixed bounding box to remove empty background,
reducing memory usage for model training.
"""

import os
import glob

import SimpleITK as sitk

# Fixed crop range (determined empirically from the training data)
CROP_RANGE = {
    "x_first": 17, "x_last": 141,
    "y_first": 29, "y_last": 200,
    "z_first": 30, "z_last": 166,
}


def crop_volume(arr, crop_range=None):
    """Crop a 3D numpy array (z, y, x) to the given range."""
    if crop_range is None:
        crop_range = CROP_RANGE
    return arr[
        crop_range["z_first"]:crop_range["z_last"],
        crop_range["y_first"]:crop_range["y_last"],
        crop_range["x_first"]:crop_range["x_last"],
    ]


def crop_nifti_files(folder_path, crop_range=None):
    """Crop all NIfTI files in *folder_path* in-place.

    Args:
        folder_path: Directory containing ``.nii`` / ``.nii.gz`` files.
        crop_range: Dict with keys ``x_first, x_last, y_first, y_last,
                     z_first, z_last``. Uses ``CROP_RANGE`` by default.
    """
    if crop_range is None:
        crop_range = CROP_RANGE

    nii_files = (
        glob.glob(os.path.join(folder_path, "*.nii"))
        + glob.glob(os.path.join(folder_path, "*.nii.gz"))
    )
    if not nii_files:
        print("No NIfTI files found.")
        return

    for fpath in nii_files:
        img = sitk.ReadImage(fpath)
        arr = sitk.GetArrayFromImage(img)

        cropped_arr = crop_volume(arr, crop_range)
        cropped_img = sitk.GetImageFromArray(cropped_arr)
        cropped_img.SetSpacing(img.GetSpacing())
        cropped_img.SetDirection(img.GetDirection())
        cropped_img.SetOrigin(img.GetOrigin())

        # Write via temp file to avoid corruption
        ext = '.nii.gz' if fpath.endswith('.nii.gz') else '.nii'
        tmp = fpath[:-len(ext)] + "_tmp" + ext
        sitk.WriteImage(cropped_img, tmp)
        os.remove(fpath)
        os.rename(tmp, fpath)
        print(f"Cropped: {os.path.basename(fpath)}")

    print(f"Done. Crop range: x=[{crop_range['x_first']}:{crop_range['x_last']}], "
          f"y=[{crop_range['y_first']}:{crop_range['y_last']}], "
          f"z=[{crop_range['z_first']}:{crop_range['z_last']}]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crop NIfTI brain volumes.")
    parser.add_argument("folder", help="Path to folder containing .nii files")
    args = parser.parse_args()
    crop_nifti_files(args.folder)
