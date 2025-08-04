from ultralytics import YOLO
import cv2
import os
import re

TRAINING = True # set to True for model training
PROJECT_NAME = 'SuctionCupMarksDetector'
MODEL_PATH = f'.\\{PROJECT_NAME}\\train2\\weights\\best.pt' # path to the model used for testing
OUTPUT_PATH = f'.\\{PROJECT_NAME}\\' # output path with predictions on test data
BASE_PATH = 'C:\\Users\\ivan_zagainov\\Downloads\\Batch3\\SuctionCupMarks' # Test the model on test samples Batch3 subfolder
SCALE_FACTOR = 25 # make image 25 times smaller to fit 640 height

def load_image_preprocess_run_model(path, image_name):
    image = cv2.imread(f'{path}/{image_name}', cv2.IMREAD_GRAYSCALE)
    if image is None:
        return
    (h, w) = image.shape[:2]
    small_image = cv2.resize(image, (int(w/SCALE_FACTOR), int(h/SCALE_FACTOR)), interpolation = cv2.INTER_AREA)
    rgb_small_image = cv2.cvtColor(small_image, cv2.COLOR_GRAY2RGB) # image for logging
    
    # Perform object detection on an image using the model
    results = model(
        rgb_small_image,
        show=False,
        save=False, # True for default logging
        save_txt=False, # True for default logging
        save_conf=False, # True for default logging
        save_crop=False, # True for default logging
        show_conf=False, # True for default logging
        show_labels=False, # True for default logging
        # line_width=1,
        conf=0.636, # this value must be updated after model training for optimal confidence threshold!!!
        rect=True, # allow padding to rectangular input images
        imgsz=[640, 320], # indicating exact image dimensions reduces inference time significantly!!!
        max_det=15 # max number of detections per image
        )
    
    # Process model results
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        for item in boxes.data:
            x0, y0, x1, y1, conf, _ = item
            cv2.rectangle(rgb_small_image, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), thickness=1) # Draw prediction    
    cv2.imwrite(OUTPUT_PATH + image_name, rgb_small_image)

if TRAINING:
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolo11n.pt')
    # Train the model using the 'coco.yaml' dataset for 250 epochs
    results = model.train(
        data='C:\\Users\\ivan_zagainov\\Downloads\\YOLO_dataset\\Suction_Cup_Mark_Detector\\coco.yaml',
        project=PROJECT_NAME,
        epochs=250,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.1,
        translate=0.1,
        scale=0.1,
        fliplr=0.5,
        flipud=0.5,
        degrees=5.0,
        # mixup=0.25,
        mosaic=0.0,
        erasing=0.0,
        auto_augment=None,
        )
    print(results)
else: # Testing
    model = YOLO(MODEL_PATH) # Load model

    # Loop through all folders
    list_folders = os.listdir(BASE_PATH)
    for idx, folder_name in enumerate(list_folders):
        path = BASE_PATH + '/' + folder_name
        print(path)
        all_files = os.listdir(path)

        # Process frontal image
        frontal_pattern = re.compile(r".*_.*_0_0_.*.bmp")
        frontal_image_name = [file for file in all_files if frontal_pattern.match(file)][0]
        print(frontal_image_name)
        load_image_preprocess_run_model(path, frontal_image_name)

        # Process rear image
        rear_pattern = re.compile(r".*_.*_3_1_.*.bmp")
        rear_image_name = [file for file in all_files if rear_pattern.match(file)][0]
        print(rear_image_name)
        load_image_preprocess_run_model(path, rear_image_name)

