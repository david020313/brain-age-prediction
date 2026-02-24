"""
Grad-CAM visualization for CNN3D brain age model.

Generates a 3D Grad-CAM heatmap showing which brain regions contribute most
to the predicted age, and saves the result as a NIfTI file.

Usage:
    python gradcam.py --model_path best_cnn3d.pth --input scan.nii.gz
"""

import argparse
import os

import numpy as np
import SimpleITK as sitk
import torch
from pytorch_grad_cam import GradCAM

from models.cnn3d import CNN3D


class RegressionTarget:
    """Grad-CAM target for regression (returns raw model output)."""

    def __init__(self):
        self.category = None

    def __call__(self, model_output):
        return model_output


def read_nifti_for_gradcam(filepath):
    """Read a NIfTI file and return (tensor, sitk_image).

    Returns:
        tensor: Shape ``(1, 1, D, H, W)`` ready for model input.
        sitk_img: Original SimpleITK image (for spacing/origin metadata).
    """
    sitk_img = sitk.ReadImage(filepath)
    arr = sitk.GetArrayFromImage(sitk_img)
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor, sitk_img


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for CNN3D brain age")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Input NIfTI file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output NIfTI file (default: <input>_gradcam.nii.gz)")
    parser.add_argument("--target_layer", type=str, default="conv6",
                        help="Target convolutional layer name (default: conv6)")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        if base.endswith('.nii'):
            base = base[:-4]
        args.output = f"{base}_gradcam.nii.gz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = CNN3D(num_classes=1).to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    # Read input
    input_tensor, sitk_img = read_nifti_for_gradcam(args.input)

    # Target layer
    target_layer = getattr(model, args.target_layer)
    targets = [RegressionTarget()]

    # Compute Grad-CAM
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # remove batch dim

    # Save as NIfTI
    out_img = sitk.GetImageFromArray(grayscale_cam)
    out_img.SetSpacing(sitk_img.GetSpacing())
    out_img.SetOrigin(sitk_img.GetOrigin())
    out_img.SetDirection(sitk_img.GetDirection())
    sitk.WriteImage(out_img, args.output)
    print(f"Grad-CAM saved to {args.output}")


if __name__ == "__main__":
    main()
