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

avg_angle = 0
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line
        # Optionally, you can filter lines by their angle:
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        avg_angle += angle
        # For instance, you can ignore extremely steep or shallow lines if needed:
        # if not (abs(angle) < 10 or abs(angle) > 80):
        #     continue
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2) # Compute the direction vector of the line
avg_angle /= len(lines)
avg_angle = 90 - avg_angle
print(avg_angle)

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

avg_intersection, all_inters = average_intersections(lines)
def compute_rectifying_homography(theta, vx, vy, output_width):
    """
    Computes a homography that rectifies the perspective so that lines
    that are tilted by angle 'theta' (in radians) and converge to vanishing
    point (vx, vy) in the original image become vertical in the rectified image.
    The output image has width 'output_width'; after transformation the vanishing
    point is horizontally centered (at output_width/2) and sent to infinity.
    
    Parameters:
      theta         - (float) The measured tilt of the lines (radians), where the 
                      lines should be vertical but are rotated by theta.
      vx, vy        - (float) The coordinates (in pixels) of the vanishing point in 
                      the original image.
      output_width  - (float) The desired width of the output image.
    
    Returns:
      H             - (3x3 numpy array) The rectifying homography.
    """
    # 1. Rotation matrix R: rotate by -theta.
    R = np.array([
        [ np.cos(-theta), -np.sin(-theta), 0],
        [ np.sin(-theta),  np.cos(-theta), 0],
        [             0,              0,  1]
    ], dtype=np.float32)
    # Note: Alternatively, using trigonometric parity:
    # R = np.array([[ np.cos(theta),  np.sin(theta), 0],
    #               [-np.sin(theta),  np.cos(theta), 0],
    #               [             0,              0,  1]], dtype=np.float32)
    
    # Apply R to the vanishing point (in homogeneous coordinates):
    V = np.array([vx, vy, 1], dtype=np.float32).reshape(3, 1)
    V_rot = R @ V  # V_rot = (vx_rot, vy_rot, 1)
    vx_rot, vy_rot = V_rot[0, 0], V_rot[1, 0]
    
    # 2. Translation matrix T: move the rotated vanishing point horizontally so that its x == output_width/2.
    T = np.array([
        [1, 0, output_width/2 - vx_rot],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    V_trans = T @ V_rot  # V_trans should be (output_width/2, vy_rot, 1)
    
    # 3. Projective transformation P: send the translated vanishing point to infinity.
    # We choose P such that P*(output_width/2, vy_rot, 1)ᵀ = (output_width/2, vy_rot, 0)
    # (Note: vy_rot should be nonzero. If vy_rot is negative because the VP is above the image,
    #  that’s acceptable.)
    if vy_rot == 0:
        raise ValueError("After rotation the vanishing point's y-coordinate is zero.")
    
    P = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, -1/ vy_rot, 1]
    ], dtype=np.float32)
    
    # Compose all transformations: H = P * T * R.
    H = P @ T @ R
    return H

# --- Example Usage ---
# Suppose your computed vanishing point is:
vx, vy = avg_intersection[0], avg_intersection[1]   # for example; note vy is negative if it lies above the image.

# Transform the image. Set the output size to cover the entire transformed content.
# (You might first want to compute the warped positions of the image corners to determine output size.)
output_size = (-width, -height)
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
