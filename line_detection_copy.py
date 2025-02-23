import cv2
import numpy as np
import math
def line_in_normal_form(x1, y1, x2, y2):
    """
    Given two endpoints (x1, y1), (x2, y2), return (nx, ny, c),
    so that nx*x + ny*y + c = 0 is the infinite line in normal form.
    The vector (nx, ny) is a unit normal to that line.
    """
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return None  # degenerate segment
    
    # Direction vector
    # We'll rotate (dx, dy) by +90° to get a normal
    nx_unnorm = -dy
    ny_unnorm =  dx
    
    # Normalize (nx, ny) to unit length
    denom = math.hypot(nx_unnorm, ny_unnorm)
    nx = nx_unnorm / denom
    ny = ny_unnorm / denom
    
    # c is computed by plugging an endpoint (x1,y1) into n·p + c = 0 => c = -n·p
    c = -(nx * x1 + ny * y1)
    return (nx, ny, c)
def lines_are_close(lineA, lineB, angle_thresh_rad, dist_thresh):
    """
    lineA, lineB: each in form (nx, ny, c)
    angle_thresh_rad: maximum angle difference
    dist_thresh: max allowed perpendicular distance between lines
    """
    nx1, ny1, c1 = lineA
    nx2, ny2, c2 = lineB
    
    # 1) Angle check
    dot = nx1*nx2 + ny1*ny2
    # Because normals might face 'opposite' ways, take absolute value
    dot = abs(dot)
    # If dot > 1 due to float precision, clamp it
    dot = min(dot, 1.0)
    angle_diff = math.acos(dot)
    
    if angle_diff > angle_thresh_rad:
        return False
    
    # 2) Distance check
    dist = abs(c2 - c1)
    if dist > dist_thresh:
        return False
    
    return True

def can_merge(seg1, seg2, angle_thresh_rad, gap_thresh):
    """
    Returns True if seg1 and seg2 are roughly collinear
    and close enough to be merged.
    """
    # Convert endpoints to numpy arrays.
    p1 = np.array([seg1[0], seg1[1]], dtype=np.float32)
    p2 = np.array([seg1[2], seg1[3]], dtype=np.float32)
    p3 = np.array([seg2[0], seg2[1]], dtype=np.float32)
    p4 = np.array([seg2[2], seg2[3]], dtype=np.float32)
    
    # Define directional vectors and compute their norms.
    v1 = p2 - p1
    v2 = p4 - p3
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return False
    
    # Compute the angles of both segments.
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    # Account for periodicity, using the minimal difference.
    diff_angle = min(abs(angle1 - angle2), np.pi - abs(angle1 - angle2))
    if diff_angle > angle_thresh_rad:
        return False

    return lines_are_close(line_in_normal_form(*seg1), line_in_normal_form(*seg2), angle_thresh_rad, gap_thresh)

def merge_two_segments(seg1, seg2, d):
    """
    Merges two segments along a common normalized direction d.
    Returns the new merged segment (using extreme projected endpoints).
    """
    pts = np.array([
        [seg1[0], seg1[1]],
        [seg1[2], seg1[3]],
        [seg2[0], seg2[1]],
        [seg2[2], seg2[3]]
    ], dtype=np.float32)
    # Project each endpoint onto the vector d.
    projections = pts.dot(d)
    min_idx = np.argmin(projections)
    max_idx = np.argmax(projections)
    return (pts[min_idx][0], pts[min_idx][1], pts[max_idx][0], pts[max_idx][1])

def process_lines(segments, angle_threshold=30, gap_threshold=60):
    """
    Merges line segments that are roughly collinear.
    
    Parameters:
      segments: list or numpy array of segments, each in form (x1, y1, x2, y2).
      angle_threshold: maximum angular difference (in degrees) for segments to merge.
      gap_threshold: maximum allowed gap (in pixel units) along the line for merging.
    
    Returns:
      A numpy array of merged segments.
    """
    # Convert segments into a list of tuples.
    if isinstance(segments, np.ndarray):
        if len(segments.shape) == 3 and segments.shape[1] == 1:
            segments = segments.reshape(-1, 4)
        segments = [tuple(s.tolist()) for s in segments]
    
    angle_thresh_rad = np.deg2rad(angle_threshold)
    merged_segments = segments[:]
    changed = True

    # Repeat merging until no further merge is found.
    while changed:
        changed = False
        new_segments = []
        used = [False] * len(merged_segments)
        
        for i in range(len(merged_segments)):
            if used[i]:
                continue
                
            seg_i = merged_segments[i]
            p1 = np.array([seg_i[0], seg_i[1]], dtype=np.float32)
            p2 = np.array([seg_i[2], seg_i[3]], dtype=np.float32)
            v = p2 - p1
            norm_v = np.linalg.norm(v)
            if norm_v == 0:
                continue
                
            # Determine the normalized direction.
            d = v / norm_v
            merged_seg = seg_i
            
            # Compare with each subsequent segment.
            for j in range(i + 1, len(merged_segments)):
                if used[j]:
                    continue
                seg_j = merged_segments[j]
                if can_merge(merged_seg, seg_j, angle_thresh_rad, gap_threshold):
                    # Merge the segments if possible.
                    merged_seg = merge_two_segments(merged_seg, seg_j, d)
                    used[j] = True
                    changed = True  # A merge has occurred.
            
            new_segments.append(merged_seg)
            used[i] = True
            
        merged_segments = new_segments
        
    return np.array(merged_segments, dtype=int)

# Optional: helper function if you want to filter by orientation.
def angle_from_vertical(angle):
    """
    Returns the absolute deviation (in degrees) from vertical.
    """
    angle = abs(angle)
    return 90 - angle

def find_lines(frame):
    """
    Finds lines in an image using Canny edge detection and HoughLinesP,
    then merges roughly collinear lines.
    
    Parameters:
      frame: The input image (assumed to have BGRA color channels).
    
    Returns:
      Tuple containing merged lines and their angles.
    """
    # Convert to grayscale.
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    # Blur to reduce noise.
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imshow("asd", edges)
    
    # Detect line segments.
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, 
                            minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        # Merge detected segments.
        lines = process_lines(lines)
        # Compute the angle of each merged line.
        angles = np.degrees(np.arctan2(
            lines[:, 3] - lines[:, 1],
            lines[:, 2] - lines[:, 0]
        ))
    
        true_lines = []
        true_angles = []
        for i in range(len(angles)):
            # Filter based on the deviation from vertical if desired.
            if angle_from_vertical(angles[i]) < 45:
                true_lines.append(lines[i])
                true_angles.append(angles[i])
        lines = np.array(true_lines)
        angles = np.array(true_angles)
    
        return lines, angles
    return None, None

# === Test / Example ===
if __name__ == "__main__":
    # Example set of segments:
    # Two horizontal segments that are close/overlapping and one diagonal segment.
    segments = [
        (10, 100, 200, 100),   # Horizontal segment
        (195, 100, 300, 100),  # Overlapping/touching horizontal segment
        (50, 150, 100, 200)    # Diagonal segment (won't merge with the horizontal ones)
    ]
    
    merged_lines = process_lines(segments, angle_threshold=10, gap_threshold=20)
    print("Merged Lines:")
    print(merged_lines)
    
    # For testing with an image, uncomment the following lines:
    # image = cv2.imread('frame.png', cv2.IMREAD_UNCHANGED)
    # # Optionally, resize: image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
    # true_lines, angles = find_lines(image)
    # line_image = image.copy()
    # for line in true_lines:
    #     cv2.line(line_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
    # cv2.imshow("Merged Lines", line_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
