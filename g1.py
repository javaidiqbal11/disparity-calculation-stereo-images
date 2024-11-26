import cv2
import numpy as np
import matplotlib.pyplot as plt



def normalize_image(img):
  """Normalizes an image using min-max normalization.

  Args:
    img: The input image.

  Returns:
    The normalized image.
  """

  img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
  return img_normalized


def hierarchical_block_matching(left_img, right_img, pyramid_levels=3, block_size=32):
    """
    Performs hierarchical block matching to estimate disparity maps.

    Args:
        left_img: Left image.
        right_img: Right image.
        pyramid_levels: Number of pyramid levels.
        block_size: Block size for matching.

    Returns:
        Disparity map.
    """

    # Ensure even dimensions for pyramid levels
    height, width = left_img.shape[:2]
    new_height = height // 2**pyramid_levels * 2**pyramid_levels
    new_width = width // 2**pyramid_levels * 2**pyramid_levels
    left_img = cv2.resize(left_img, (new_width, new_height))
    right_img = cv2.resize(right_img, (new_width, new_height))

    # Create image pyramids
    left_pyramid = [left_img]
    right_pyramid = [right_img]
    for _ in range(pyramid_levels - 1):
        left_pyramid.append(cv2.pyrDown(left_pyramid[-1]))
        right_pyramid.append(cv2.pyrDown(right_pyramid[-1]))

    # Initialize disparity map
    disparity_map = np.zeros_like(left_img, dtype=np.float32)

    # Iterate over pyramid levels
    for level in range(pyramid_levels - 1, -1, -1):
        curr_left_img = left_pyramid[level]
        curr_right_img = right_pyramid[level]

        # Compute disparity map at current level
        for y in range(0, curr_left_img.shape[0] - block_size, block_size):
            for x in range(0, curr_left_img.shape[1] - block_size, block_size):
                block = curr_left_img[y:y+block_size, x:x+block_size]
                min_sad = float('inf')
                best_disp = 0
                for disp in range(-block_size//2, block_size//2 + 1):
                    if x + disp < 0 or x + disp + block_size >= curr_right_img.shape[1]:
                        continue
                    curr_block = curr_right_img[y:y+block_size, x+disp:x+disp+block_size]
                    sad = np.sum(np.abs(block - curr_block))
                    if sad < min_sad:
                        min_sad = sad
                        best_disp = disp
                disparity_map[y:y+block_size, x:x+block_size] = best_disp * (2 ** level)

    return disparity_map



# Load stereo images

left_img = normalize_image(cv2.imread("CG images/CG images/left.png", 0))
right_img = normalize_image(cv2.imread("CG images/CG images/right.png", 0))
# left_img = cv2.imread("real images/real images/left.png", 0)
# right_img = cv2.imread("real images/real images/right.png", 0)

# Compute disparity map
disparity_map = hierarchical_block_matching(left_img, right_img)

ret, thresh = cv2.threshold(left_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Threshold the disparity map
# thresholded_map = cv2.threshold(disparity_map, thresh, 255, cv2.THRESH_BINARY)[1]

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(disparity_map, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the result
# cv2.imshow('Disparity Map with Bounding Boxes', disparity_map)
# cv2.imwrite("disparity_map.png", disparity_map)

# Normalize the disparity map (adjust the range as needed)
disparity_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

# Check if the normalized map is valid
if np.max(disparity_normalized) > 0:
    # Save the normalized disparity map as a PNG image
    cv2.imwrite('disparity_map.png', disparity_normalized)
else:
    print("Disparity map is empty or all values are zero.")
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Min-Max Normalization (to 8-bit range)
# disparity_normalized_minmax = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
#
# # Z-Score Normalization (adjust mean and std_dev as needed)
# mean = np.mean(disparity_map)
# std_dev = np.std(disparity_map)
# disparity_normalized_zscore = (disparity_map - mean) / std_dev
#
# cv2.imwrite('disparity_normalized_minmax.png', disparity_normalized_minmax)
# cv2.imwrite('disparity_normalized_zscore.png', disparity_normalized_zscore)
# Update the color disparity map using the new grayscale disparity values

# # Visualize the disparity map
# plt.figure(figsize=(10, 8))
# plt.imshow(disparity_map, cmap='viridis')
# plt.colorbar()
# plt.title('Disparity Map')
# plt.axis('off')
# plt.show()
