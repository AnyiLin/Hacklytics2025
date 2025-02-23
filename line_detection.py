import cv2
import numpy as np
import math

def line_in_normal_form(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return None
    
    # Direction vector
    # Rotate (dx, dy) by +90° to get a normal
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
    nx1, ny1, c1 = lineA
    nx2, ny2, c2 = lineB
    
    dot = nx1*nx2 + ny1*ny2
    dot = abs(dot)
    dot = min(dot, 1.0)
    angle_diff = math.acos(dot)
    
    if angle_diff > angle_thresh_rad:
        return False
    
    dist = abs(c2 - c1)
    if dist > dist_thresh:
        return False
    
    return True

def can_merge(seg1, seg2, angle_thresh_rad, gap_thresh):
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
    
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    
    diff_angle = min(abs(angle1 - angle2), np.pi - abs(angle1 - angle2))
    if diff_angle > angle_thresh_rad:
        return False

    return lines_are_close(line_in_normal_form(*seg1), line_in_normal_form(*seg2), angle_thresh_rad, gap_thresh)

def merge_two_segments(seg1, seg2, d):
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
    if isinstance(segments, np.ndarray):
        if len(segments.shape) == 3 and segments.shape[1] == 1:
            segments = segments.reshape(-1, 4)
        segments = [tuple(s.tolist()) for s in segments]
    
    angle_thresh_rad = np.deg2rad(angle_threshold)
    merged_segments = segments[:]
    changed = True

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
            
        merged_segments = new_segments
        
    return np.array(merged_segments, dtype=int)

def angle_from_vertical(angle):
    angle = abs(angle)
    return 90 - angle

def find_lines(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, 
                            minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        lines = process_lines(lines)
        angles = np.degrees(np.arctan2(
            lines[:, 3] - lines[:, 1],
            lines[:, 2] - lines[:, 0]
        ))
    
        true_lines = []
        true_angles = []
        for i in range(len(angles)):
            if angle_from_vertical(angles[i]) < 45:
                true_lines.append(lines[i])
                true_angles.append(angles[i])
        lines = np.array(true_lines)
        angles = np.array(true_angles)
    
        return lines, angles
    return None, None
