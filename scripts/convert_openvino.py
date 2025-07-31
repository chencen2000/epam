import os
import argparse
import torch
from openvino import convert_model
from openvino.runtime import save_model
from src.models.unet import UNet  

def convert_modified_unet_to_openvino(
    pth_path: str,
    input_shape: tuple,
    onnx_path: str,
    openvino_output_dir: str,
    n_channels: int = 1,
    n_classes: int = 4,
    bilinear: bool = False
):
    """
    Converts a modified UNet PyTorch model to OpenVINO IR format.
    """
    print("[INFO] Initializing model...")
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear, architecture="modified")

    print("[INFO] Loading weights...")
    checkpoint = torch.load(pth_path, map_location='cpu')
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print("[INFO] Exporting to ONNX...")
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11
    )
    print(f"[INFO] ONNX model saved to: {onnx_path}")

    print("[INFO] Converting ONNX to OpenVINO IR...")
    openvino_model = convert_model(onnx_path)
    os.makedirs(openvino_output_dir, exist_ok=True)
    save_model(
        openvino_model,
        output_model=os.path.join(openvino_output_dir, "unet_openvino.xml")
    )
    print(f"[INFO] OpenVINO IR model saved to: {openvino_output_dir}/unet_openvino.xml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert UNet PyTorch model to OpenVINO format")
    parser.add_argument("--pth_path", type=str, default="../models/light_weight_mix_data/best_model.pth", help="Path to .pth checkpoint file")
    parser.add_argument("--input_shape", type=int, nargs=4, default=(1, 1, 1792, 1792), help="Model input shape as (B, C, H, W)")
    parser.add_argument("--onnx_path", type=str, default="model.onnx", help="Path to save ONNX model")
    parser.add_argument("--openvino_output_dir", type=str, default="../models/openvino_ir", help="Directory to save OpenVINO IR model")
    parser.add_argument("--n_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--n_classes", type=int, default=4, help="Number of output classes")
    parser.add_argument("--bilinear", action="store_true",  help="Use bilinear upsampling")

    args = parser.parse_args()
    
    convert_modified_unet_to_openvino(
        pth_path=args.pth_path,
        input_shape=tuple(args.input_shape),
        onnx_path=args.onnx_path,
        openvino_output_dir=args.openvino_output_dir,
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        bilinear=args.bilinear
    )
