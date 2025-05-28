import matplotlib.pyplot as plt
import random
import numpy as np
import cv2

SQUARE_MASK_SIZE = 500

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
    left = min(polygon, key=lambda x: x[0])[0]
    bottom = min(polygon, key=lambda x: x[1])[1]
    right = max(polygon, key=lambda x: x[0])[0]
    top = max(polygon, key=lambda x: x[1])[1]
    return (left, bottom, right, top)

def normalize(polygon):
    left, bottom, right, top = find_bounding_box(polygon)
    # normalize each point in the polygon
    return [((x-left) / (right-left) * SQUARE_MASK_SIZE, (y-bottom) / (top-bottom) * SQUARE_MASK_SIZE) for x, y in polygon]

# Define the initial square (clockwise order)
square = [(0, 0), (1, 0), (1, 1), (0, 1)]

# Add vertices and distort them
distorted_polygon = add_vertices(square, num_vertices_per_side=5, max_distortion=0.1)

# Close the polygon by appending the first point at the end
distorted_polygon.append(distorted_polygon[0])

normalized_polygon = normalize(distorted_polygon)

# Plot the original square and the distorted polygon
plt.figure(figsize=(6, 6))
plt.plot(*zip(*[(x*SQUARE_MASK_SIZE, y*SQUARE_MASK_SIZE) for x, y in square] + [square[0]*SQUARE_MASK_SIZE]), label="Original Square", linestyle="--", color="blue")
plt.plot(*zip(*normalized_polygon), label="Distorted Polygon", color="red")
plt.scatter(*zip(*normalized_polygon), color="red", s=10)  # Mark vertices
plt.legend()
plt.axis("equal")
plt.title("Distorted Polygon from Square")
plt.show()

image_width = SQUARE_MASK_SIZE
image_height = SQUARE_MASK_SIZE
mask = np.zeros((image_height, image_width), dtype=np.uint8)

# Define the vertices of a polygon
polygon_vertices = np.array(normalized_polygon, dtype=np.int32)

# Draw filled polygons
cv2.fillPoly(mask, [polygon_vertices], color=(255)) 

# Display the image with the filled polygons
cv2.imshow('Filled Polygons', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()