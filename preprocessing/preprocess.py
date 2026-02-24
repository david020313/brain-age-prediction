"""
MRI Preprocessing Pipeline.

Complete workflow for preparing raw T1-weighted MRI scans:
    1. ANTsPy skull stripping (via antspynet brain_extraction)
    2. Rigid registration to MNI152 template
    3. SPM tissue segmentation via MATLAB (produces c1/c2 maps)

Requirements:
    - ANTsPy & antspynet (Python)
    - MATLAB with SPM12 (called via subprocess)
"""

import os
import glob
import time
import tempfile
import subprocess

import ants
from antspynet.utilities import brain_extraction


# ============================================================================
# N4 Bias Field Correction
# ============================================================================

def n4_bias_field_correction(image, mask=None, shrink_factor=4,
                             convergence=None, **kwargs):
    """Wrapper around ANTsPy N4 bias field correction."""
    if convergence is None:
        convergence = {"iters": [50, 50, 50, 50], "tol": 1e-7}
    return ants.n4_bias_field_correction(
        image, mask=mask, shrink_factor=shrink_factor,
        convergence=convergence, **kwargs
    )


def abp_n4(image, intensity_truncation=(0.025, 0.975, 256), mask=None,
           usen3=False, verbose=False):
    """Truncate intensities and apply N4 bias field correction."""
    outimage = ants.iMath(
        image, "TruncateIntensity",
        intensity_truncation[0], intensity_truncation[1],
        intensity_truncation[2]
    )
    if usen3:
        outimage = ants.n3_bias_field_correction(outimage, 4)
        outimage = ants.n3_bias_field_correction(outimage, 2)
    else:
        outimage = n4_bias_field_correction(outimage, mask, verbose=verbose)
    return outimage


# ============================================================================
# Core Processing
# ============================================================================

def process_nii_file(nii_file_path, base_output_dir, matlab_script_dir,
                     template_image):
    """Run the complete preprocessing pipeline on a single NIfTI file.

    Steps:
        1. Read and reorient image
        2. Skull stripping (ANTsPy deep learning)
        3. Rigid registration to MNI template
        4. SPM segmentation via MATLAB → c1 (GM) and c2 (WM)

    Args:
        nii_file_path: Path to the raw .nii / .nii.gz file.
        base_output_dir: Root output directory.
        matlab_script_dir: Directory containing ``run_spm_seg.m``.
        template_image: Pre-loaded ANTsPy MNI template image.
    """
    try:
        base_name = os.path.basename(nii_file_path).replace('.nii.gz', '').replace('.nii', '')
        subject_output_dir = os.path.join(base_output_dir, base_name)
        os.makedirs(subject_output_dir, exist_ok=True)
        print(f"    - Output: {subject_output_dir}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Read
            print("    - Reading image ...")
            raw_img = ants.image_read(nii_file_path, reorient='IAL')

            # Skull stripping
            print("    - Skull stripping ...")
            prob_mask = brain_extraction(raw_img, modality='t1', verbose=False)
            brain_mask = ants.get_mask(prob_mask, low_thresh=0.5)
            masked = ants.mask_image(raw_img, brain_mask)

            # Registration
            print("    - Rigid registration ...")
            reg = ants.registration(
                fixed=template_image, moving=masked,
                type_of_transform='Rigid', verbose=False
            )
            registered = reg['warpedmovout']

            # Save for MATLAB
            reg_path = os.path.join(tmpdir, 'registered_for_matlab.nii')
            registered.to_file(reg_path)

            # SPM segmentation
            print("    - SPM segmentation (MATLAB) ...")
            matlab_cmd = (
                f"cd('{matlab_script_dir}'); "
                f"run_spm_seg('{reg_path}', '{subject_output_dir}', '{base_name}'); "
                f"exit;"
            )
            subprocess.run(
                ['matlab', '-wait', '-nosplash', '-nodesktop', '-r', matlab_cmd],
                shell=True, capture_output=True, text=True
            )

        print(f"    - ✅ {base_name} done.")
    except Exception as e:
        print(f"    - ❌ Error processing {nii_file_path}: {e}")


# ============================================================================
# Batch Processing Entry Point
# ============================================================================

def batch_preprocess(source_folder, output_dir, matlab_script_dir, template_path):
    """Process all NIfTI files in *source_folder*.

    Args:
        source_folder: Directory containing raw .nii / .nii.gz files.
        output_dir: Root output directory.
        matlab_script_dir: Directory containing ``run_spm_seg.m``.
        template_path: Path to MNI152 template .nii file.
    """
    nii_files = (
        glob.glob(os.path.join(source_folder, '*.nii'))
        + glob.glob(os.path.join(source_folder, '*.nii.gz'))
    )

    if not nii_files:
        print("No .nii / .nii.gz files found in source folder.")
        return

    print(f"Found {len(nii_files)} files. Loading template ...")
    template = ants.image_read(template_path, reorient='IAL')

    total_start = time.time()
    for i, fpath in enumerate(nii_files):
        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{len(nii_files)}: {os.path.basename(fpath)}")
        print(f"{'='*60}")
        t0 = time.time()
        process_nii_file(fpath, output_dir, matlab_script_dir, template)
        print(f"    - Time: {time.time() - t0:.1f}s")

    print(f"\n🎉 All done! Total: {(time.time() - total_start)/60:.1f} min.")


if __name__ == "__main__":
    # ---- EDIT THESE PATHS ----
    SOURCE_FOLDER = r"C:\Users\user\Desktop\AD nii"
    BASE_OUTPUT_DIR = r"C:\Users\user\Desktop\AD nii_preprocessed"
    MATLAB_SCRIPT_DIR = os.path.join(os.path.dirname(__file__))  # same dir
    TEMPLATE_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'assets',
        'mni_icbm152_t1_tal_nlin_sym_09a.nii'
    )
    batch_preprocess(SOURCE_FOLDER, BASE_OUTPUT_DIR, MATLAB_SCRIPT_DIR, TEMPLATE_PATH)
