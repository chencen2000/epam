import os
import random

import numpy as np
import cv2


def load_image_color(image_path):
    if not os.path.exists(image_path): return None
    return cv2.imread(image_path, cv2.IMREAD_COLOR)


def load_image_grayscale(image_path):
    if not os.path.exists(image_path): return None
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def save_image(image, path):
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
    return cv2.imwrite(path, image)


def get_margin_bbox(margins, img_w, img_h):
    b_xmin = int(img_w * (margins["left"] / 100.0))
    b_ymin = int(img_h * (margins["top"] / 100.0))
    b_xmax = int(img_w * (1.0 - margins["right"] / 100.0))
    b_ymax = int(img_h * (1.0 - margins["bottom"] / 100.0))

    return b_xmin, b_ymin, b_xmax, b_ymax



def apply_scale(image, scale, interpolation=cv2.INTER_CUBIC):
    """
    Apply scaling with cubic interpolation by default
    """
    if scale <= 0: return image, image.shape[1], image.shape[0]
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    if new_width <= 0 or new_height <= 0: return None, 0, 0
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation), new_width, new_height


def apply_rotation(image, angle, interpolation=cv2.INTER_CUBIC, border_value=0):
    """
    Apply rotation with cubic interpolation by default
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    ones = np.ones(shape=(len(corners), 1))
    points_ones = np.hstack([corners, ones])
    transformed_points = M.dot(points_ones.T).T
    
    x_coords, y_coords = transformed_points[:,0], transformed_points[:,1]
    nW_robust = int(np.ceil(x_coords.max() - x_coords.min()))
    nH_robust = int(np.ceil(y_coords.max() - y_coords.min()))

    M[0,2] += (nW_robust / 2) - center[0]
    M[1,2] += (nH_robust / 2) - center[1]
    
    rotated_image = cv2.warpAffine(image, M, (nW_robust, nH_robust), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    return rotated_image, nW_robust, nH_robust


def check_overlap(new_bbox, placed_bboxes):
    for placed_bbox in placed_bboxes:
        if (new_bbox[0] < placed_bbox[2] and new_bbox[2] > placed_bbox[0] and
            new_bbox[1] < placed_bbox[3] and new_bbox[3] > placed_bbox[1]):
            return True
    return False


def distort_point(point, max_distortion=0.1):
    """Randomly distort a point within a given range."""
    x, y = point
    x += random.uniform(-max_distortion, max_distortion)
    y += random.uniform(-max_distortion, max_distortion)
    return (x, y)

def add_vertices(square, num_vertices_per_side=3, max_distortion=0.1):
    """Add vertices to the sides of a square and distort them."""
    new_polygon = []
    for i in range(len(square)):
        start = square[i]
        end = square[(i + 1) % len(square)]
        
        # Add the starting vertex
        new_polygon.append(distort_point(start, max_distortion))
        
        # Add intermediate vertices
        for j in range(1, num_vertices_per_side + 1):
            t = j / (num_vertices_per_side + 1)
            intermediate_point = (
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1])
            )
            new_polygon.append(distort_point(intermediate_point, max_distortion))
    
    return new_polygon

def find_bounding_box(polygon):
    """Find bounding box of polygon."""
    left = min(polygon, key=lambda x: x[0])[0]
    bottom = min(polygon, key=lambda x: x[1])[1]
    right = max(polygon, key=lambda x: x[0])[0]
    top = max(polygon, key=lambda x: x[1])[1]
    return (left, bottom, right, top)

def normalize_polygon_to_size(polygon, target_width, target_height):
    """Normalize polygon to fit within target dimensions."""
    left, bottom, right, top = find_bounding_box(polygon)
    # normalize each point in the polygon
    return [((x-left) / (right-left) * target_width, (y-bottom) / (top-bottom) * target_height) for x, y in polygon]

def create_distorted_mask_boundary(mask_shape, num_vertices_per_side=3, max_distortion=0.1):
    """
    Create a distorted polygon boundary mask from the original rectangular mask.
    Returns the distorted mask.
    """
    height, width = mask_shape
    
    # Define the initial square (normalized coordinates)
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    
    # Add vertices and distort them
    distorted_polygon = add_vertices(square, num_vertices_per_side=num_vertices_per_side, max_distortion=max_distortion)
    
    # Normalize to mask dimensions
    normalized_polygon = normalize_polygon_to_size(distorted_polygon, width, height)
    
    # Create distorted boundary mask
    distorted_boundary_mask = np.zeros((height, width), dtype=np.uint8)
    polygon_vertices = np.array(normalized_polygon, dtype=np.int32)
    cv2.fillPoly(distorted_boundary_mask, [polygon_vertices], color=255)
    
    return distorted_boundary_mask

def apply_distorted_boundary_to_mask(original_mask, num_vertices_per_side=3, max_distortion=0.1):
    """
    Apply distorted polygon boundary to an existing mask.
    """
    # Create distorted boundary mask
    distorted_boundary = create_distorted_mask_boundary(
        original_mask.shape, 
        num_vertices_per_side=num_vertices_per_side, 
        max_distortion=max_distortion
    )
    
    # Apply the distorted boundary to the original mask
    distorted_mask = cv2.bitwise_and(original_mask, distorted_boundary)
    
    return distorted_mask

