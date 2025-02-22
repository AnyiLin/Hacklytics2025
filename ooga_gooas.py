import cv2
import numpy as np

# Load the image
image = cv2.imread('frame.png')
width = image.shape[1]
height = image.shape[0]
width = int(width / 3)
height = int(height / 3)
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# Example: assume you have computed the vanishing point V = (v_x, v_y, 1)
# For demonstration, let’s pick a dummy value (in practice, compute this from your lines)
v_x, v_y = -500000, 5000000000000  # adjust these based on your image

# Compute alpha so that alpha * v_x + 1 = 0, sending the vanishing point to infinity
alpha = -1.0 / v_x

# Construct the homography matrix H.
H = np.array([[1, 0, 0],
              [0, 1, 0],
              [alpha, 0, 1]], dtype=np.float32)

# Transform the image. Set the output size to cover the entire transformed content.
# (You might first want to compute the warped positions of the image corners to determine output size.)
output_size = (width, height)
warped = cv2.warpPerspective(image, H, output_size,
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))

cv2.imshow("Rectified Image", warped)
cv2.imshow('Detected Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()