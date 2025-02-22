import cv2
import numpy as np

# Load the image
image = cv2.imread('frame.png')
width = image.shape[1]
height = image.shape[0]
width = int(width / 3)
height = int(height / 3)
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# Assume you computed source points from your slanted lines.
# These might only cover the field region.
src_pts = np.float32([
    [120, 80],    # Example: top-left point (from detected intersections)
    [520, 70],    # Example: top-right point
    [540, 400],   # Example: bottom-right point
    [100, 420]    # Example: bottom-left point
])

# Destination points for your chosen rectangle (for the area of interest)
dst_pts = np.float32([
    [0, 0],
    [800, 0],
    [800, 600],
    [0, 600]
])

# Step 1: Compute the homography
H = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Step 2: Get the original image dimensions
h, w = image.shape[:2]
orig_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

# Step 3: Warp the corners of the entire image using H
warped_corners = cv2.perspectiveTransform(orig_corners, H)
warped_corners = warped_corners.reshape(-1, 2)

# Step 4: Compute the bounding box around the warped corners
min_x = warped_corners[:, 0].min()
min_y = warped_corners[:, 1].min()
max_x = warped_corners[:, 0].max()
max_y = warped_corners[:, 1].max()

# Step 5: Create a translation matrix to shift the warped image to positive coordinates
T = np.array([[1, 0, -min_x],
              [0, 1, -min_y],
              [0, 0, 1]], dtype=np.float32)
H_adjusted = T @ H

# Step 6: Define the output image size and warp the entire image
output_width = int(max_x - min_x)
output_height = int(max_y - min_y)

warped_image = cv2.warpPerspective(image, H_adjusted, (output_width, output_height),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,  # Fill extra areas with black.
                                   borderValue=(0, 0, 0))

width = warped_image.shape[1]
height = warped_image.shape[0]
width = int(width / 3)
height = int(height / 3)
warped_image = cv2.resize(warped_image, (width, height), interpolation=cv2.INTER_AREA)
# Display the result
cv2.imshow("Entire Rectified Image", warped_image)
cv2.imshow('Detected Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()