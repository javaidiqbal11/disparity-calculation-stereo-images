import csv

import cv2
import numpy as np
from skimage.measure import find_contours

block_sizes = [32, 16, 8]
ambiguity_threshold = 10  # Difference between first and second lowest MSE
mse_threshold = 50  # Acceptable MSE threshold


def compute_mse(left_image, right_image, block_size=32, max_disparity=64):
    height, width = left_image.shape
    mse_values = np.full((height, width), np.inf)  # Initialize MSE map with infinity

    # Loop through the left image
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Extract block from left image
            left_block = left_image[y:y + block_size, x:x + block_size]

            # Skip if block is incomplete
            if left_block.shape != (block_size, block_size):
                continue

            # Search in the right image
            best_mse = float('inf')
            for d in range(max_disparity):  # Search range for disparity
                x_right = x - d  # Adjust x-coordinate in the right image

                if x_right < 0:
                    break  # Out of bounds

                # Extract block from the right image
                right_block = right_image[y:y + block_size, x_right:x_right + block_size]

                # Skip if block is incomplete
                if right_block.shape != (block_size, block_size):
                    continue

                # Calculate MSE between the blocks
                mse = np.mean((left_block.astype(np.float32) - right_block.astype(np.float32)) ** 2)

                # Keep track of the minimum MSE for the current block
                best_mse = min(best_mse, mse)

            # Store the best MSE value in the MSE map
            mse_values[y:y + block_size, x:x + block_size] = best_mse
    mask = ~np.isinf(mse_values).any(axis=1)
    filtered_array = mse_values[mask]

    return filtered_array


def compute_disparity_with_uncertainty(left_img, right_img, max_disparity):
    height, width = left_img.shape
    disparity_map = np.zeros((height, width), dtype=np.float32)
    uncertainty_map = np.zeros((height, width), dtype=np.uint8)


    for block_size in block_sizes:
        step = block_size // 2
        for y in range(0, height, step):
            for x in range(0, width, step):
                min_mse = float('inf')
                second_min_mse = float('inf')
                best_disparity = 0

                # Search range
                for d in range(0, max_disparity):
                    if x - d < 0:  # Ensure valid indices
                        continue

                    block_left = left_img[y:y + block_size, x:x + block_size]
                    block_right = right_img[y:y + block_size, (x - d):(x - d) + block_size]

                    if block_left.shape != block_right.shape or block_left.size == 0:
                        continue

                    mse = np.mean((block_left - block_right) ** 2)
                    if mse < min_mse:
                        second_min_mse = min_mse
                        min_mse = mse
                        best_disparity = d
                    elif mse < second_min_mse:
                        second_min_mse = mse

                # Mark unconfident matches
                if min_mse > mse_threshold or (second_min_mse - min_mse) < ambiguity_threshold:
                    uncertainty_map[y:y + block_size, x:x + block_size] = 1
                disparity_map[y:y + block_size, x:x + block_size] = best_disparity

    disparity_map = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite('grayscale_disparity.png', disparity_map)
    hsv_disparity = cv2.applyColorMap(disparity_map, cv2.COLORMAP_HSV)
    cv2.imwrite('hsv_disparity.png', hsv_disparity)

    return disparity_map, uncertainty_map


def generate_contours(uncertainty_map):
    # Use skimage's find_contours for Marching Squares
    contours = find_contours(uncertainty_map, level=0.5)  # Contours at binary threshold
    return contours


def extract_bounding_boxes_and_properties(contours, image):
    properties = []
    for contour in contours:
        # Compute bounding box
        x, y, w, h = cv2.boundingRect(np.array(contour, dtype=np.int32))
        area = w * h

        # Extract region of interest (ROI)
        roi = image[y:y + h, x:x + w]

        # Calculate average color
        avg_color = np.mean(roi)

        # Store properties
        properties.append({
            'position': (x, y),
            'width': w,
            'height': h,
            'area': area,
            'avg_color': avg_color
        })
    return properties


def match_bounding_boxes(left_props, right_props, position_weight=1.0, size_weight=1.0, color_weight=1.0):
    matches = []
    for left_box in left_props:
        best_match = None
        best_score = float('inf')

        for right_box in right_props:
            # Compute differences
            pos_diff = np.linalg.norm(np.array(left_box['position']) - np.array(right_box['position']))
            size_diff = abs(left_box['width'] - right_box['width']) + abs(left_box['height'] - right_box['height'])
            area_diff = abs(left_box['area'] - right_box['area'])
            color_diff = abs(left_box['avg_color'] - right_box['avg_color'])

            # Compute match score
            score = (position_weight * pos_diff +
                     size_weight * (size_diff + area_diff) +
                     color_weight * color_diff)

            if score < best_score:
                best_score = score
                best_match = right_box

        matches.append((left_box, best_match, best_score))
    return matches


# Load images and compute uncertainty maps
left_image = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)

# Generate contours for both images
ldisparity, left_uncertainty = compute_disparity_with_uncertainty(left_image, right_image, max_disparity=64)
rdisparity, right_uncertainty = compute_disparity_with_uncertainty(right_image, left_image, max_disparity=64)

binary_confidence_left = (left_uncertainty < mse_threshold).astype(np.uint8) * 255
binary_confidence_right = (right_uncertainty < mse_threshold).astype(np.uint8) * 255

cv2.imwrite('binary_confidence_left.png', binary_confidence_left)
cv2.imwrite('binary_confidence_right.png', binary_confidence_right)

left_contours = find_contours(left_uncertainty, level=0.5)
right_contours = find_contours(right_uncertainty, level=0.5)

# Extract bounding boxes and properties
left_properties = extract_bounding_boxes_and_properties(left_contours, left_image)
right_properties = extract_bounding_boxes_and_properties(right_contours, right_image)

# Match bounding boxes
matches = match_bounding_boxes(left_properties, right_properties)

# Visualization (optional)
left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
right_image_rgb = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)

for left_box, right_box, score in matches:
    if right_box is not None:
        # Draw matched bounding boxes
        x1, y1 = left_box['position']
        w1, h1 = left_box['width'], left_box['height']
        cv2.rectangle(left_image_rgb, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

        x2, y2 = right_box['position']
        w2, h2 = right_box['width'], right_box['height']
        cv2.rectangle(right_image_rgb, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

# Display results
cv2.imshow('Left Image Matches', left_image_rgb)
cv2.imshow('Right Image Matches', right_image_rgb)
cv2.imwrite('left_matches.png', left_image_rgb)
cv2.imwrite('right_matches.png', right_image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()


def compute_disparity_from_bounding_boxes(left_properties, right_properties, disparity_map):
    refined_disparity_map = disparity_map.copy()
    height, width = disparity_map.shape

    for left_box, right_box, score in matches:
        if right_box is not None:
            # Calculate disparity from bounding box centers
            x_left, y_left = left_box['position']
            w_left, h_left = left_box['width'], left_box['height']
            center_left = (x_left + w_left // 2, y_left + h_left // 2)

            x_right, y_right = right_box['position']
            w_right, h_right = right_box['width'], right_box['height']
            center_right = (x_right + w_right // 2, y_right + h_right // 2)

            disparity = center_left[0] - center_right[0]

            # Update disparity map within the bounding box
            for y in range(y_left, y_left + h_left):
                for x in range(x_left, x_left + w_left):
                    if 0 <= y < height and 0 <= x < width:
                        refined_disparity_map[y, x] = disparity

    return refined_disparity_map


# Compute bounding box disparities and refine disparity map
refined_disparity_map = compute_disparity_from_bounding_boxes(
    left_properties, right_properties, disparity_map=ldisparity
)

# Normalize refined disparity map for visualization
refined_disparity_map = cv2.normalize(refined_disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display and save the refined disparity map
cv2.imshow('Refined Disparity Map', refined_disparity_map)
cv2.imwrite('refined_disparity_map.png', refined_disparity_map)
cv2.waitKey(0)
cv2.destroyAllWindows()

from skimage.measure import find_contours


def marching_squares_map(uncertainty_map, image):
    contours = find_contours(uncertainty_map, level=0.5)
    marching_squares_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for contour in contours:
        pts = np.array(contour, dtype=np.int32)
        cv2.polylines(marching_squares_image, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

    return marching_squares_image




marching_left = marching_squares_map(left_uncertainty, left_image)
marching_right = marching_squares_map(right_uncertainty, right_image)

cv2.imwrite('marching_squares_left.png', marching_left)
cv2.imwrite('marching_squares_right.png', marching_right)


def bounding_box_map(image, properties):
    bounding_box_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for prop in properties:
        x, y = prop['position']
        w, h = prop['width'], prop['height']
        cv2.rectangle(bounding_box_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return bounding_box_image

bounding_left = bounding_box_map(left_image, left_properties)
bounding_right = bounding_box_map(right_image, right_properties)

cv2.imwrite('bounding_boxes_left.png', bounding_left)
cv2.imwrite('bounding_boxes_right.png', bounding_right)


# Grayscale version
updated_grayscale_disparity = cv2.normalize(refined_disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite('updated_grayscale_disparity.png', updated_grayscale_disparity)

# HSV version
updated_hsv_disparity = cv2.applyColorMap(updated_grayscale_disparity, cv2.COLORMAP_HSV)
cv2.imwrite('updated_hsv_disparity.png', updated_hsv_disparity)


# Example usage
left_image = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)

mse_values = compute_mse(left_image, right_image)
mse_map_normalized = cv2.normalize(mse_values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite('mse_map.png', mse_map_normalized)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.hist(mse_values.flatten(), bins=50, color='blue', edgecolor='black')
plt.title('MSE Histogram')
plt.savefig('mse_histogram.png')  # Save histogram


def save_block_matching_details_to_csv(block_details, filename):
    # Define CSV headers
    headers = ['x', 'y', 'mse', 'disparity', 'std_dev', 'search_directions']

    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(block_details)


# Data Files-------------------------------------

def compute_block_matching_details(left_image, right_image, block_size=32, max_disparity=64):
    height, width = left_image.shape
    block_details = []

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            left_block = left_image[y:y + block_size, x:x + block_size]

            if left_block.shape != (block_size, block_size):
                continue

            best_mse = float('inf')
            best_disparity = 0
            search_directions = []

            std_dev = np.std(left_block)  # Standard deviation of the block

            for d in range(max_disparity):
                x_right = x - d
                if x_right < 0:
                    break

                search_directions.append('left')
                right_block = right_image[y:y + block_size, x_right:x_right + block_size]
                if right_block.shape != (block_size, block_size):
                    continue

                mse = np.mean((left_block.astype(np.float32) - right_block.astype(np.float32)) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_disparity = d

            # Append block details
            block_details.append({
                'x': x,
                'y': y,
                'mse': best_mse,
                'disparity': best_disparity,
                'std_dev': std_dev,
                'search_directions': ', '.join(set(search_directions))
            })

    return block_details


# Example usage
left_image = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)

block_details = compute_block_matching_details(left_image, right_image)
save_block_matching_details_to_csv(block_details, 'block_matching_details.csv')



def save_bounding_box_details_to_csv(bounding_boxes, filename):
    headers = ['x', 'y', 'width', 'height', 'area', 'avg_color', 'position_diff', 'size_diff', 'color_diff', 'disparity']

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(bounding_boxes)


def compute_bounding_box_details(contours_left, contours_right, left_image, right_image):
    bounding_boxes = []

    # Compute bounding box properties and matches
    for left_box, right_box in zip(contours_left, contours_right):
        if len(left_box) == 0 or len(right_box) == 0:
            continue
        left_contour = np.array(left_box, dtype=np.int32)
        right_contour = np.array(right_box, dtype=np.int32)
        x_left, y_left, w_left, h_left = cv2.boundingRect(left_contour)
        x_right, y_right, w_right, h_right = cv2.boundingRect(right_contour)

        avg_color_left = np.mean(left_image[y_left:y_left+h_left, x_left:x_left+w_left])
        avg_color_right = np.mean(right_image[y_right:y_right+h_right, x_right:x_right+w_right])
        color_diff = np.abs(avg_color_left - avg_color_right)

        position_diff = np.sqrt((x_left - x_right) ** 2 + (y_left - y_right) ** 2)
        size_diff = np.abs((w_left * h_left) - (w_right * h_right))
        disparity = x_left + w_left // 2 - (x_right + w_right // 2)

        bounding_boxes.append({
            'x': x_left,
            'y': y_left,
            'width': w_left,
            'height': h_left,
            'area': w_left * h_left,
            'avg_color': avg_color_left,
            'position_diff': position_diff,
            'size_diff': size_diff,
            'color_diff': color_diff,
            'disparity': disparity
        })

    return bounding_boxes

bounding_boxes = compute_bounding_box_details(left_contours, right_contours, left_image, right_image)
save_bounding_box_details_to_csv(bounding_boxes, 'bounding_box_details.csv')
