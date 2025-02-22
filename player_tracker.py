import cv2
import numpy as np
import math

image = cv2.imread('frame.png', cv2.IMREAD_UNCHANGED)
width = image.shape[1]
height = image.shape[0]
width = int(width / 3)
height = int(height / 3)
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
lines = cv2.HoughLinesP(edges, 
                        rho=1, 
                        theta=np.pi/180, 
                        threshold=50, 
                        minLineLength=100, 
                        maxLineGap=10)
# Create a copy of the original image to draw the detected lines
line_image = image.copy()

if lines is not None:
    avg_angle = 0
    left_most = None
    right_most = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if left_most is None or x1 < left_most[0]:
            left_most = (x1, y1, x2, y2)
        if right_most is None or x2 > right_most[2]:
            right_most = (x1, y1, x2, y2)
        # Optionally, you can filter lines by their angle:
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        # For instance, you can ignore extremely steep or shallow lines if needed:
        # if not (abs(angle) < 10 or abs(angle) > 80):
        #     continue
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2) # Compute the direction vector of the line
        
def line_from_points(p1, p2):
    """
    Convert two points into a homogeneous line representation.
    The line is given by: a*x + b*y + c = 0  as [a, b, c].
    """
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return np.array([a, b, c], dtype=float)

def intersection_of_lines(L1, L2, eps=1e-6):
    """
    Compute the intersection of two lines in homogeneous form using the cross product.
    Returns None if the lines are nearly parallel.
    """
    # Compute cross product:
    X = np.cross(L1, L2)
    if abs(X[2]) < eps:
        # Lines are nearly parallel; no reliable intersection.
        return None
    # Convert to Cartesian coordinates:
    x = X[0] / X[2]
    y = X[1] / X[2]
    return np.array([x, y])

def average_intersections(line_segments):
    """
    Given a list of line segments (each as (x1, y1, x2, y2)),
    convert them to lines, compute pairwise intersections,
    and return the average intersection point.
    """
    # Convert each segment into its homogeneous line representation.
    lines = [line_from_points((seg[0], seg[1]), (seg[2], seg[3])) for seg in line_segments]
    
    intersections = []
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            pt = intersection_of_lines(lines[i], lines[j])
            if pt is not None:
                intersections.append(pt)
                
    if intersections:
        intersections = np.array(intersections)
        avg_pt = np.mean(intersections, axis=0)
        return avg_pt, intersections
    else:
        return None, None

# Example usage:
line_segments = [
    (100, 200, 150, 50),   # segment 1
    (110, 210, 160, 60),   # segment 2
    (90, 190, 140, 45)     # segment 3
    # You can add more segments as needed.
]

avg_intersection, all_inters = average_intersections(line_segments)
# Example: assume you have computed the vanishing point V = (v_x, v_y, 1)
# For demonstration, let’s pick a dummy value (in practice, compute this from your lines)
v_x, v_y = avg_intersection[0], avg_intersection[1]  # adjust these based on your image

# Compute alpha so that alpha * v_x + 1 = 0, sending the vanishing point to infinity
alpha = -1.0 / v_x

# Construct the homography matrix H.
H = np.array([[1, 0, 0],
              [0, 1, 0],
              [alpha, 0, 1]], dtype=np.float32)

# Transform the image. Set the output size to cover the entire transformed content.
# (You might first want to compute the warped positions of the image corners to determine output size.)
output_size = (width*20, height*20)
warped = cv2.warpPerspective(image, H, output_size,
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
warped = cv2.resize(warped, (width, height), interpolation=cv2.INTER_AREA)


# Display the original image, the edges, and the image with detected lines
cv2.imshow("Original Image", image)
cv2.imshow("Edges", edges)
cv2.imshow("Detected Lines", line_image)
cv2.imshow("Blurred", blurred)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
