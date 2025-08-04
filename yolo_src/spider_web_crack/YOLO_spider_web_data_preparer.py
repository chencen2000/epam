import os
import cv2
import re
import numpy as np
import time
import random

IM_SIZE = 512 # best result, but unstable training = 320 pix
# path to all samples with cracks / all negative samples
BASE_PATH = 'C:\\Users\\ivan_zagainov\\Downloads\\YOLO_dataset\\SpiderWeb512\\crack_samples'
output_path = 'C:\\Users\\ivan_zagainov\\Downloads\\YOLO_dataset\\SpiderWeb512\\negative_patches_dirt' # where to store cropped patches

# helper function
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

def crop_resize_transpose_save(image, image_name, file_suffix):
    # resize and save
    resized = cv2.resize(image, (IM_SIZE, IM_SIZE), interpolation = cv2.INTER_AREA)
    cv2.imwrite(f'{output_path}/{image_name.split('.')[0]}_{file_suffix}.png', resized) # write small image
    # transpose and save (optional data augmentation)
    resized = cv2.transpose(resized)
    cv2.imwrite(f'{output_path}/{image_name.split('.')[0]}_{file_suffix}_t.png', resized) # write transposed small image
    
    # make 5 random 0.75 crops (optional data augmentation)
    crops_path = output_path + '/random_crops'
    (h, w) = image.shape[:2]
    for iter in range(0, 5):
        x0 = random.randint(0, w // 4) # random offset range
        y0 = random.randint(0, h // 4) # random offset range
        size = int(w * 0.75)
        print('W =', w, 'H =', h, 'x0 =', x0, 'y0 =', y0, size, 'x', size)
        cropped_image = image[y0:y0+size, x0:x0+size]
        resized = cv2.resize(cropped_image, (IM_SIZE, IM_SIZE), interpolation = cv2.INTER_AREA)
        cv2.imwrite(f'{crops_path}/{image_name.split('.')[0]}_{file_suffix}_rnd{iter}.png', resized) # write cropped image
        # transpose and save
        resized = cv2.transpose(resized)
        cv2.imwrite(f'{crops_path}/{image_name.split('.')[0]}_{file_suffix}_rnd{iter}_t.png', resized) # write transposed cropped image
    
    return

if __name__ == "__main__":
    all_files = os.listdir(BASE_PATH)
    pattern = re.compile(r".*.bmp") # uses all frontal and back samples from the folder
    images = [file for file in all_files if pattern.match(file)]
    for image_name in images:
        print(image_name)
        path = BASE_PATH

        '''
        # this is alternative code to above version to loop through image subfolders in some Batch folder
    BASE_PATH = 'C:\\Users\\ivan_zagainov\\Downloads\\Batch1\\Dirt'

    # Loop through all folders
    list_folders = os.listdir(BASE_PATH)
    for idx, folder_name in enumerate(list_folders):
        path = BASE_PATH + '/' + folder_name
        print(path)
        all_files = os.listdir(path)
        pattern = re.compile(r".*_.*_3_1_.*.bmp") # or use r".*_.*_0_0_.*.bmp" pattern for frontal surface
        images = [file for file in all_files if pattern.match(file)]
        image_name = images[0]
        print(image_name)
        '''

        # Load image
        image = cv2.imread(f'{path}/{image_name}', cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        # Image for drawing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Detect device region
        start_time = time.time()
        x, y, w, h = get_device_bbox(image, 0.04)
        detect_time = time.time()
        cv2.rectangle(rgb_image, (x,y), (x+w,y+h), (0,255,0), thickness=15) # draw
            
        # Define overall region for Spider Web Crack classification
        rx = x + int(w * 0.025)
        rw = int(w * 0.95)
        ry = y + int(h * 0.01)
        rh = int(h * 0.98)
        # alternative - use full crop
        #rx = x
        #rw = w
        #ry = y
        #rh = h
        cv2.rectangle(rgb_image, (rx,ry), (rx+rw,ry+rh), (255,0,0), thickness=15) # draw
        
        # top
        cropped_image = image[ry:ry+rw, rx:rx+rw]
        crop_resize_transpose_save(cropped_image, image_name, 'top')
        # middle
        cropped_image = image[int(ry+rh/2-rw/2):int(ry+rh/2+rw/2), rx:rx+rw]
        crop_resize_transpose_save(cropped_image, image_name, 'mid')
        # bottom
        cropped_image = image[ry+rh-rw:ry+rh, rx:rx+rw]
        crop_resize_transpose_save(cropped_image, image_name, 'bot')

        # bbox_detection_time = round((detect_time - start_time) * 1000, 2) # in milliseconds
        # print(f'Bbox detection time = {bbox_detection_time}ms')
            
        resized_rgb = cv2.resize(rgb_image, None, fx=0.2, fy=0.2) # downscale log image to save space           
        cv2.imwrite(f'{output_path}/logs/{image_name.split('.')[0]}_log.png', resized_rgb) # write log image