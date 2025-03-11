import argparse

import torch
from ultralytics import YOLO  # Correct way to load YOLOv8 models


def export_onnx(model_path, output_onnx, img_size, opset_version):
    # Load trained YOLO model
    model = YOLO(model_path)  # Corrected from torch.hub.load

    # Define a dummy input (1 batch, 3 channels, img_size x img_size image size)
    dummy_input = torch.randn(1, 3, img_size, img_size)

    # Export to ONNX
    torch.onnx.export(
        model.model,  # Access the PyTorch model inside YOLO class
        dummy_input,
        output_onnx,
        opset_version=opset_version,
        input_names=["images"],
        output_names=["outputs"],
    )

    print(f"ONNX model saved as {output_onnx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO model to ONNX format")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the trained .pt model"
    )
    parser.add_argument(
        "--output-onnx", type=str, default="model.onnx", help="Output ONNX file name"
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="Image size for dummy input"
    )
    parser.add_argument(
        "--opset-version", type=int, default=11, help="ONNX opset version"
    )

    args = parser.parse_args()
    export_onnx(args.model_path, args.output_onnx, args.img_size, args.opset_version)
