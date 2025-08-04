import numpy as np
import cv2
import time
import argparse
import os
from openvino.runtime import Core


def run_openvino_inference(
    model_path_xml: str,
    image_path: str,
    resize_to: tuple = (1792, 1792),
    normalize: bool = True,
    apply_argmax: bool = True
):
    """
    Runs inference on a single image using OpenVINO IR model.
    """
    # 1. Load model
    core = Core()
    model = core.read_model(model=model_path_xml)
    compiled_model = core.compile_model(model, device_name="CPU")

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # 2. Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Grayscale input
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.resize(img, resize_to)

    if normalize:
        img = img.astype(np.float32) / 255.0

    input_tensor = np.expand_dims(np.expand_dims(img, axis=0), axis=0).astype(np.float32)  # (1, 1, H, W)

    # 3. Run inference
    start_time = time.time()
    output = compiled_model([input_tensor])[output_layer]
    end_time = time.time()

    print(f"[INFO] Inference time: {end_time - start_time:.4f} seconds")

    # 4. Postprocess
    output = np.squeeze(output)
    if apply_argmax:
        output = np.argmax(output, axis=0)

    return output


def save_output_image(output_array: np.ndarray, output_path: str):
    """
    Saves a prediction mask to an image file.
    """
    if output_array.ndim == 2:
        output_vis = (output_array * (255 // output_array.max())).astype(np.uint8)
    else:
        raise ValueError("Expected output shape (H, W) after argmax")

    cv2.imwrite(output_path, output_vis)
    print(f"[INFO] Saved output image to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenVINO inference on image and save output.")
    parser.add_argument("--model_xml", required=True, help="Path to OpenVINO IR .xml file")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--output_path", required=True, help="Path to save output image")
    parser.add_argument("--resize", nargs=2, type=int, default=(1792, 1792), help="Resize image to (H W)")
    args = parser.parse_args()

    output_mask = run_openvino_inference(
        model_path_xml=args.model_xml,
        image_path=args.image_path,
        resize_to=tuple(args.resize)
    )

    save_output_image(output_mask, args.output_path)


# python inference_openvino.py \
#     --model_xml ./openvino_ir/unet_openvino.xml \
#     --image_path ./sample_input.png \
#     --output_path ./predicted_mask.png \
#     --resize 1792 1792

