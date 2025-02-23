import cv2
import numpy as np
import math

def can_merge(seg1, seg2, angle_thresh_rad, gap_thresh):
    # Pre-allocate arrays for better memory usage
    p1 = np.array([seg1[0], seg1[1]], dtype=np.float32)
    p2 = np.array([seg1[2], seg1[3]], dtype=np.float32)
    p3 = np.array([seg2[0], seg2[1]], dtype=np.float32)
    p4 = np.array([seg2[2], seg2[3]], dtype=np.float32)
    
    v1 = p2 - p1
    v2 = p4 - p3
    # Compute norms once
    norm1 = np.linalg.norm(v1)
    if norm1 == 0:
        return False
    norm2 = np.linalg.norm(v2)
    if norm2 == 0:
        return False
    
    # Calculate angles and differences in one go
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    diff_angle = min(abs(angle1 - angle2), np.pi - abs(angle1 - angle2))
    if diff_angle > angle_thresh_rad:
        return False

    # Normalize direction vector once
    d = v1 / norm1
    
    # Compute all projections at once
    projs = np.array([np.dot(p, d) for p in [p1, p2, p3, p4]])
    interval1 = [min(projs[0], projs[1]), max(projs[0], projs[1])]
    interval2 = [min(projs[2], projs[3]), max(projs[2], projs[3])]
    
    if interval1[1] < interval2[0] - gap_thresh or interval2[1] < interval1[0] - gap_thresh:
        return False
    
    # Compute perpendicular distances directly
    vec3, vec4 = p3 - p1, p4 - p1
    proj3, proj4 = np.dot(vec3, d), np.dot(vec4, d)
    dist3 = np.linalg.norm(vec3 - proj3 * d)
    dist4 = np.linalg.norm(vec4 - proj4 * d)
    
    return not (dist3 > gap_thresh or dist4 > gap_thresh)

def process_lines(segments, angle_threshold=10, gap_threshold=20):
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
                    # Inline merge_two_segments functionality
                    pts = np.array([
                        [merged_seg[0], merged_seg[1]],
                        [merged_seg[2], merged_seg[3]],
                        [seg_j[0], seg_j[1]],
                        [seg_j[2], seg_j[3]]
                    ], dtype=np.float32)
                    projections = pts.dot(d)
                    min_idx, max_idx = np.argmin(projections), np.argmax(projections)
                    merged_seg = (pts[min_idx][0], pts[min_idx][1], 
                                pts[max_idx][0], pts[max_idx][1])
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

    # Optimized line detection
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, 
                            minLineLength=100, maxLineGap=10)

    # Process lines with minimal conversions
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

        # # Draw lines efficiently
        # line_image = frame.copy()
        # for line in true_lines:
        #     cv2.line(line_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)

        # cv2.imshow("Detected Lines", line_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return lines, angles
    return None, None


# # Optimize image processing pipeline
# image = cv2.imread('frame.png', cv2.IMREAD_UNCHANGED)
# width, height = image.shape[1] // 2, image.shape[0] // 2  # Integer division
# image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
# true_lines, avg_angle = find_lines(image)
# # Draw lines efficiently
# line_image = image.copy()
# for line in true_lines:
#     cv2.line(line_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)

# cv2.imshow("Detected Lines", line_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()