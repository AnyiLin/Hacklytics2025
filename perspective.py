import cv2
import numpy as np
import math

image = cv2.imread('frame.png', cv2.IMREAD_UNCHANGED)
width = image.shape[1]
height = image.shape[0]
width = int(width / 2)
height = int(height / 2)
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
def can_merge(seg1, seg2, angle_thresh_rad, gap_thresh):
    """
    Determine if seg1 and seg2 can be merged.
    seg1 and seg2 are each tuples: (x1, y1, x2, y2).
    They are mergeable if:
      - Their orientations (angles) differ by less than angle_thresh_rad.
      - Their endpoints, when projected onto the common direction,
        yield intervals that overlap (or nearly) and
      - The perpendicular distance from seg2's endpoints to seg1's line is below gap_thresh.
    """
    # Convert endpoints to numpy arrays
    p1 = np.array([seg1[0], seg1[1]], dtype=np.float32)
    p2 = np.array([seg1[2], seg1[3]], dtype=np.float32)
    p3 = np.array([seg2[0], seg2[1]], dtype=np.float32)
    p4 = np.array([seg2[2], seg2[3]], dtype=np.float32)
    
    v1 = p2 - p1
    v2 = p4 - p3
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return False
    
    # Compute the angles of the segments.
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    
    # Normalize the angular difference to [0, pi/2]
    diff_angle = np.abs(angle1 - angle2)
    diff_angle = min(diff_angle, np.pi - diff_angle)
    if diff_angle > angle_thresh_rad:
        return False

    # Use seg1's direction as the reference.
    d = v1 / norm1

    # Project the endpoints of both segments onto d.
    proj1 = np.dot(p1, d)
    proj2 = np.dot(p2, d)
    interval1 = [min(proj1, proj2), max(proj1, proj2)]
    
    proj3 = np.dot(p3, d)
    proj4 = np.dot(p4, d)
    interval2 = [min(proj3, proj4), max(proj3, proj4)]
    
    # Check if the projection intervals overlap (or nearly so).
    if interval1[1] < interval2[0] - gap_thresh or interval2[1] < interval1[0] - gap_thresh:
        return False
    
    # Compute the perpendicular distances of seg2's endpoints to seg1's line.
    def perp_distance(pt, line_pt, d):
        vec = pt - line_pt
        proj_length = np.dot(vec, d)
        proj_vec = proj_length * d
        diff = vec - proj_vec
        return np.linalg.norm(diff)
    
    dist_p3 = perp_distance(p3, p1, d)
    dist_p4 = perp_distance(p4, p1, d)
    
    if dist_p3 > gap_thresh or dist_p4 > gap_thresh:
        return False

    return True

def merge_two_segments(seg1, seg2, d):
    """
    Merge two line segments seg1 and seg2 that are mergeable.
    The merged segment will span the extremes of both segments when projected along d.
    """
    p1 = np.array([seg1[0], seg1[1]], dtype=np.float32)
    p2 = np.array([seg1[2], seg1[3]], dtype=np.float32)
    p3 = np.array([seg2[0], seg2[1]], dtype=np.float32)
    p4 = np.array([seg2[2], seg2[3]], dtype=np.float32)
    
    pts = np.array([p1, p2, p3, p4])
    # Project points onto d.
    projections = pts.dot(d)
    min_idx = np.argmin(projections)
    max_idx = np.argmax(projections)
    new_seg = (pts[min_idx][0], pts[min_idx][1], pts[max_idx][0], pts[max_idx][1])
    return new_seg

def merge_segments(segments, angle_threshold=10, gap_threshold=20):
    """
    Merge line segments that are nearly collinear and close together.

    Parameters:
      segments: NumPy array with shape (N, 1, 4) or (N, 4), where each segment is (x1, y1, x2, y2)
      angle_threshold: maximum difference in orientation (in degrees) allowed for merging.
      gap_threshold: maximum allowed gap (in pixels) between segments for merging.

    Returns:
      A list of merged segments in the form [(x1, y1, x2, y2), ...]
    """
    # If segments are in NumPy array form, convert to a list of tuples.
    if isinstance(segments, np.ndarray):
        if len(segments.shape) == 3 and segments.shape[1] == 1:
            segments = segments.reshape(-1, 4)
        segments = [tuple(s.tolist()) for s in segments]
    
    angle_thresh_rad = np.deg2rad(angle_threshold)
    merged_segments = segments[:]  # Make a copy
    changed = True

    # Iteratively merge segments until no more merges occur.
    while changed:
        changed = False
        new_segments = []
        used = [False] * len(merged_segments)
        
        for i in range(len(merged_segments)):
            if used[i]:
                continue
            seg_i = merged_segments[i]
            # Compute seg_i's direction.
            p1 = np.array([seg_i[0], seg_i[1]], dtype=np.float32)
            p2 = np.array([seg_i[2], seg_i[3]], dtype=np.float32)
            v = p2 - p1
            norm_v = np.linalg.norm(v)
            if norm_v == 0:
                continue
            d = v / norm_v
            
            merged_seg = seg_i

            for j in range(i + 1, len(merged_segments)):
                if used[j]:
                    continue
                seg_j = merged_segments[j]
                if can_merge(merged_seg, seg_j, angle_thresh_rad, gap_threshold):
                    merged_seg = merge_two_segments(merged_seg, seg_j, d)
                    used[j] = True
                    changed = True
            
            new_segments.append(merged_seg)
            used[i] = True
        merged_segments = np.copy(new_segments)
    return merged_segments

lines = merge_segments(lines)
true_lines = []
for i in range(len(lines)):
    true_lines.append([])
    for j in range(len(lines[i])):
        true_lines[i].append(int(lines[i][j]))
lines = true_lines

# Create a copy of the original image to draw the detected lines
line_image = image.copy()

angles = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line
        # Optionally, you can filter lines by their angle:
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        angles.append(angle)
        # For instance, you can ignore extremely steep or shallow lines if needed:
        # if not (abs(angle) < 10 or abs(angle) > 80):
        #     continue
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2) # Compute the direction vector of the line

med_angle = np.median(angles)
avg_angle = 0
angle_count = 0
for i in angles:
    if not abs(i - med_angle) > med_angle * 0.1:
        print(i)
        avg_angle += i
        angle_count += 1
avg_angle /= angle_count
print(avg_angle)
avg_angle = -(90 - avg_angle)
print(avg_angle)

# Get image dimensions (height and width)
(h, w) = image.shape[:2]

# Calculate the center of the image
center = (w // 2, h // 2)

# Generate the rotation matrix. The scale factor is set to 1.0 (no scaling)
M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
# Perform the rotation using warpAffine. The output image size is set as (w, h)
rotated = cv2.warpAffine(image, M, (w, h))

# Display the original image, the edges, and the image with detected lines
cv2.imshow("Original Image", image)
cv2.imshow("Edges", edges)
cv2.imshow("Detected Lines", line_image)
cv2.imshow("Blurred", blurred)
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
