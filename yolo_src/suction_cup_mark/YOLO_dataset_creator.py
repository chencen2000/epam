from pycocotools.coco import COCO
import cv2
import os

TARGET_CATEGORY_NAME = 'Suction Cup Marks' # or any other category, like 'Sticker-labels', names from coco.json annotation files
YOLO_DATASET_CLASS_NUMBER = 0 # class number for generated YOLO dataset
BASE_PATH = 'C:\\Users\\ivan_zagainov\\Downloads\\Batch1\\Suction Cup Marks\\' # path to labelled dataset (containing coco.json files)
TARGET_DATA_PATH = 'C:\\Users\\ivan_zagainov\\Downloads\\YOLO_dataset\\Suction_Cup_Mark_Detector\\' # where to store preprocessed images and labels

def convert_annotations(annotation_file_path, target_path):
    # Load the COCO annotations
    coco = COCO(annotation_file_path)

    # Access categories
    categories = coco.loadCats(coco.getCatIds())
    print("Categories:", [cat['name'] for cat in categories])
    category_list = [cat for cat in categories if cat['name']==TARGET_CATEGORY_NAME] # find corresponding category
    if not category_list: # not found in this folder!
        return
    target_category = category_list[0]
    target_id = target_category['id'] # target ID for labels

    # Access images
    image_ids = coco.getImgIds()
    images = coco.loadImgs(image_ids)
    print(f"Number of images: {len(images)}")

    # Access annotations for a specific image
    for image_id in image_ids:
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        image_name = images[image_id-1]['file_name']
        # print(f"Annotations for image {image_name}: {annotations}")

        # Load corresponding image
        image_path = annotation_file_path.removesuffix('coco.json') + image_name
        image = cv2.imread(image_path, cv2.IMREAD_COLOR_BGR)

        # Create small copy of image for YOLO dataset
        (h, w) = image.shape[:2]
        scale_factor = 25 # 25 times smaller to fit 640 height
        small_image = cv2.resize(image, (int(w/scale_factor), int(h/scale_factor)), interpolation = cv2.INTER_AREA)
        rgb_image = small_image.copy() # log image for drawing labels

        txt_path = target_path + image_name.removesuffix(".bmp") + '.txt'

        with open(txt_path, "w") as file:
            # Load annotations for it
            number_of_annotations = 0
            for annotation in annotations:
                if annotation['category_id'] != target_id: # check if the label type matches requred
                    continue
                x, y, width, height = annotation['bbox']
                small_x, small_y, small_w, small_h = round(x/scale_factor), round(y/scale_factor), round(width/scale_factor), round(height/scale_factor)
      
                # Write label annotation to YOLO txt file
                center_x = float(small_x + small_w/2.0) / int(w/scale_factor)
                center_y = float(small_y + small_h/2.0) / int(h/scale_factor)
                w_norm = float(small_w) / int(w/scale_factor)
                h_norm = float(small_h) / int(h/scale_factor)
                txt_annotation = str(YOLO_DATASET_CLASS_NUMBER) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' + str(w_norm) + ' ' + str(h_norm) + '\n'
                file.write(txt_annotation)            
                number_of_annotations += 1
            
                # Draw label on log image
                cv2.rectangle(rgb_image, (small_x,small_y), (small_x+small_w,small_y+small_h), (0, 0, 255), thickness=1)

        if number_of_annotations > 0:
            # Save small image to file
            img_path = target_path + image_name
            cv2.imwrite(img_path, small_image)

            # Save the log image with annotations drawn to the same folder with '_labels' suffix and '.png' extension
            label_image_path = img_path.removesuffix('.bmp') + '_labels.png'
            cv2.imwrite(label_image_path, rgb_image)
        else:
            os.remove(txt_path) # no annotations were found, do not need that file

if __name__ == "__main__":
    if not os.path.exists(TARGET_DATA_PATH):
        os.makedirs(TARGET_DATA_PATH)

    # Loop through all folders
    list_folders = os.listdir(BASE_PATH)
    for idx, folder_name in enumerate(list_folders):
        path = BASE_PATH + folder_name + '/' + 'coco.json'
        if os.path.exists(path):
            convert_annotations(path, TARGET_DATA_PATH)
