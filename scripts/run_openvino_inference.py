import numpy as np
import cv2
import time
import argparse
from openvino.runtime import Core, Model


def run_openvino_inference(
    model_path_xml: str,
    image_path: str,
    resize_to: tuple = (1792, 1792),
    normalize: bool = True,
    apply_argmax: bool = True,
    remove_last_two_layers: bool = False
) -> np.ndarray:
    """
    Runs inference on a single image using OpenVINO IR model, with optional removal of last 2 layers.
    Returns the output mask array.
    """
    core = Core()
    model = core.read_model(model=model_path_xml)

    if remove_last_two_layers:
        print("[INFO] Removing last 2 layers...")

        last_output_port = model.outputs[0]
        last_node = last_output_port.get_node()

        # Step 1: Go back one layer
        prev_output_port_1 = last_node.input(0).get_source_output()
        prev_node_1 = prev_output_port_1.get_node()
        print(f"[INFO] Last node removed: {last_node.get_friendly_name()} ({last_node.get_type_name()})")
        print(f"[INFO] 1st previous node: {prev_node_1.get_friendly_name()} ({prev_node_1.get_type_name()})")

        # Step 2: Go back another layer
        prev_output_port_2 = prev_node_1.input(0).get_source_output()
        prev_node_2 = prev_output_port_2.get_node()
        print(f"[INFO] 2nd previous node: {prev_node_2.get_friendly_name()} ({prev_node_2.get_type_name()})")

        # Rebuild model with earlier output
        model = Model([prev_output_port_2], model.get_parameters(), "model_without_last_2_layers")

    compiled_model = core.compile_model(model, device_name="CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.resize(img, resize_to)

    if normalize:
        img = img.astype(np.float32) / 255.0

    input_tensor = np.expand_dims(np.expand_dims(img, axis=0), axis=0).astype(np.float32)  # Shape: (1, 1, H, W)

    # Run inference
    start_time = time.time()
    for i in range(1,100):
        infer_request = compiled_model.create_infer_request()
        infer_request.infer({input_layer.any_name: input_tensor})
        output = infer_request.get_output_tensor(output_layer.index).data
    end_time = time.time()

    print(f"[INFO] Inference time: { (end_time - start_time)/100:.4f} seconds")

    # Postprocess output
    output = np.squeeze(output)
    print(f"[INFO] Output shape after squeeze: {output.shape}")

    if apply_argmax:
        output = np.argmax(output, axis=0)
        print(f"[INFO] Output shape after argmax: {output.shape}")

    return output


def save_output_image(output_array: np.ndarray, output_path: str):
    """
    Saves a prediction mask to an image file.
    """
    if output_array.ndim == 2:
        # Normalize mask to 0–255 for visualization
        if output_array.max() > 0:
            output_vis = (output_array * (255 // output_array.max())).astype(np.uint8)
        else:
            output_vis = output_array.astype(np.uint8)
        cv2.imwrite(output_path, output_vis)
        print(f"[INFO] Saved output image to {output_path}")
    else:
        raise ValueError("Expected output shape (H, W) after argmax")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenVINO inference on image and save output.")
    parser.add_argument("--model_xml", required=True, help="Path to OpenVINO IR .xml file")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--output_path", required=True, help="Path to save output image")
    parser.add_argument("--resize", nargs=2, type=int, default=(1792, 1792), help="Resize image to (H W)")
    parser.add_argument("--remove_last_two_layers", action="store_true", help="Remove the last two layers of the model")
    parser.add_argument("--no_argmax", action="store_true", help="Skip argmax on output")

    args = parser.parse_args()

    output_mask = run_openvino_inference(
        model_path_xml=args.model_xml,
        image_path=args.image_path,
        resize_to=tuple(args.resize),
        apply_argmax=not args.no_argmax,
        remove_last_two_layers=args.remove_last_two_layers
    )

    save_output_image(output_mask, args.output_path)


#     python inference_openvino.py \
#   --model_xml ./models/openvino_ir/unet.xml \
#   --image_path ./input.png \
#   --output_path ./output_mask.png \
#   --resize 1792 1792 \
#   --remove_last_two_layers

