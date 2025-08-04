from ultralytics import YOLO
import os
import cv2
import re
import numpy as np
import time


# Main script params section - please change root folders accordingly!!!
IMG_SIZE = 512 # tile size
PROJECT_NAME = 'SpiderWeb512' # subfolder to store experiments
DATASET_PATH = f'C:\\Users\\ivan_zagainov\\Downloads\\YOLO_dataset\\{PROJECT_NAME}'
EXPERIMENT_NAME = 'train3' # which experiment to use for testing
TRAINING = False # set to True to re-train model, if False - only testing is performed
MODEL_FOR_TESTING = 'last' # or best, if last - the model from the last epoch will be used for testing
MODEL_PATH = f'.\\{PROJECT_NAME}\\{EXPERIMENT_NAME}\\weights\\{MODEL_FOR_TESTING}.pt' # do not modify!!!
BASE_PATH = f'C:\\Users\\ivan_zagainov\\Downloads\\YOLO_dataset\\{PROJECT_NAME}\\hi_res_crack_samples' # positive class folder path
BATCH_PATH = 'C:\\Users\\ivan_zagainov\\Downloads\\Batch3' # negative class batch folder path
TEST_ON_NEGATIVE = True # test on all (negative) samples from a specified batch - some true positives can be found!!!
TEST_ON_REAR_SURFACE = True # if false - test will be performed on front surfaces
DEVICE_SURFACE_FILE_NAME_TEMPLATE = r".*_.*_0_0_.*.bmp" if TEST_ON_REAR_SURFACE is False else r".*_.*_3_1_.*.bmp" # do not modify!!!
SURFACE = '_front' if TEST_ON_REAR_SURFACE is False else '_rear' # do not modify!!!
if TEST_ON_NEGATIVE is not True: # do not modify!!!
    SURFACE = '' # do not modify!!!
TXT_FILE_PREFIX = 'negative_' if TEST_ON_NEGATIVE is True else 'positive_' # do not modify!!!
STATISTICS_TXT_PATH = f'.\\data\\{PROJECT_NAME}\\{TXT_FILE_PREFIX}Batch3_class_confidences_{EXPERIMENT_NAME}_{MODEL_FOR_TESTING}{SURFACE}.txt' # output txt log path


# Helper function
def get_device_bbox(image, scale):
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    filtered = cv2.medianBlur(resized, 13) # to remove bright stripes from PowerON samples, glares, etc.
    normalized = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Calculate the histogram
    hist = cv2.calcHist([normalized], [0], None, [256], [0, 256])
    # Calculate the cumulative histogram
    cumulative_hist = np.cumsum(hist)
    max_val = cumulative_hist.max()
    # Normalize the cumulative histogram
    cumulative_hist_normalized = cumulative_hist / max_val

    # Auto threshold adjustment
    threshold = 64 # suitable for most of the cases, except some rear surfaces
    if cumulative_hist_normalized[threshold] > 0.62: # area detected with this threshold is less than 38% of the image
        while cumulative_hist_normalized[threshold] > 0.62: # loop until area above threshold becomes at least 38%
            threshold -= 1
        if threshold > 40: # if it is still above 40
            threshold = max(40, threshold-4) # safe to make it even lower unless it is abobe 40 (there has to be plateau)
    # print(threshold)

    # Threshold normalized image to binary
    _, binary_image = cv2.threshold(normalized, threshold, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return int(x/scale), int(y/scale), int(w/scale), int(h/scale)
        
    # Default fall-off (if binarization didn't produce anything, which is weird)
    h, w = image.shape
    xmin = int(w * 0.12)
    ymin = int(h * 0.15)
    xmax = int(w * (1.0 - 0.12))
    ymax = int(h * (1.0 - 0.15))
    return xmin, ymin, xmax-xmin, ymax-ymin


if TRAINING:
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolo11n-cls.pt")
    # Train the model using the 'coco.yaml' dataset for 25 epochs
    results = model.train(
        data=DATASET_PATH,
        epochs=25,
        patience=0,
        batch=16,
        imgsz=IMG_SIZE,
        project=PROJECT_NAME,
        pretrained=True,
        dropout=0.5, # 0.5
        plots=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        flipud=0.5,
        degrees=5.0,
        shear=0.0,
        mixup=0.25, # 0.25???
        cutmix=0.25, # 0.25???
        mosaic=0.0,
        erasing=0.15,
        auto_augment='augmix',
        )
    print(results)
else:
    model = YOLO(MODEL_PATH)


def preprocess_and_run_classifier(cropped_image, size):
    resized = cv2.resize(cropped_image, (size, size), interpolation = cv2.INTER_AREA)
    results = model(resized)
    confidence = 0
    for result in results:
        confidence = float(result.probs.data[0])
    return confidence


def process_image(image_path, txt_file):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return -1.0, -1.0, -1.0 # error
    
    # Detect device region
    x, y, w, h = get_device_bbox(image, 0.04)
    # Define overall region for Spider Web Crack classification
    rx = x + int(w * 0.025)
    rw = int(w * 0.95)
    ry = y + int(h * 0.01)
    rh = int(h * 0.98)

    # top area
    cropped_image = image[ry:ry+rw, rx:rx+rw]
    top_conf = preprocess_and_run_classifier(cropped_image, IMG_SIZE)
    # middle area
    cropped_image = image[int(ry+rh/2-rw/2):int(ry+rh/2+rw/2), rx:rx+rw]
    mid_conf = preprocess_and_run_classifier(cropped_image, IMG_SIZE)
    # bottom area
    cropped_image = image[ry+rh-rw:ry+rh, rx:rx+rw]
    bot_conf = preprocess_and_run_classifier(cropped_image, IMG_SIZE)

    confidence = round(max(top_conf, mid_conf, bot_conf), 3)    
    top_conf_str = str(round(top_conf, 3)) + ' '
    mid_conf_str = str(round(mid_conf, 3)) + ' '
    bot_conf_str = str(round(bot_conf, 3)) + ' '

    print('Spider Web Crack Confidences = ', top_conf_str, mid_conf_str, bot_conf_str, str(confidence))
    txt_file.write(path + '/' + image_name + ' ' + top_conf_str + mid_conf_str + bot_conf_str + str(confidence) + '\n')
    return confidence


# Run the model on test samples
with open(STATISTICS_TXT_PATH, "w") as txt_file:
    total_count = 0
    detected_count = 0

    if TEST_ON_NEGATIVE is not True:
        # Loop through all files in single folder (positive class samples)
        all_files = os.listdir(BASE_PATH)
        pattern = re.compile(r".*.bmp")
        images = [file for file in all_files if pattern.match(file)]
        for image_name in images:
            print(image_name)
            path = BASE_PATH
            total_count += 1
            if process_image(f'{path}/{image_name}', txt_file) > 0.5:
                detected_count += 1
    else: # Test on negative samples from Batch
        # Loop through all folders
        list_folders = os.listdir(BATCH_PATH)
        for idx, folder_name in enumerate(list_folders):
            base_folder_path = BATCH_PATH + '/' + folder_name
            all_subfolders = os.listdir(base_folder_path)
            for idx, subfolder_name in enumerate(all_subfolders):
                path = base_folder_path + '/' + subfolder_name
                print(path)
                all_files = os.listdir(path)
                pattern = re.compile(DEVICE_SURFACE_FILE_NAME_TEMPLATE)
                images = [file for file in all_files if pattern.match(file)]
                image_name = images[0]
                print(image_name)                
                total_count += 1
                if process_image(f'{path}/{image_name}', txt_file) > 0.5:
                    detected_count += 1

    print('Total count: ', str(total_count))
    print('Detected count: ', str(detected_count))
    print('Detection ratio: ', str(round(detected_count * 100.0 / total_count, 2)), '%')
