import cv2
import numpy as np
from line_detection import find_lines
from yolo10_detection import player_detection

def line_to_abd(x1, y1, x2, y2):
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    norm = np.sqrt(A**2 + B**2) + 1e-8  # Avoid division by zero
    return (A / norm, B / norm, C / norm)

def compute_x_intercept(line_eq):
    A, B, C = line_eq
    if abs(A) < 1e-8:
        # For horizontal lines, the x-intercept is not defined.
        return None
    return -C / A

def point_line_distance(line, point):
    A, B, C = line
    x0, y0 = point
    return abs(A*x0 + B*y0 + C) / np.sqrt(A**2 + B**2 + 1e-8)

def find_two_lines_closest_to_center(lines, center):
    scored_lines = []
    for pts in lines:
        x1, y1, x2, y2 = pts
        vec_line = line_to_abd(x1, y1, x2, y2)
        d = point_line_distance(vec_line, center)
        scored_lines.append((d, pts))
    scored_lines.sort(key=lambda x: x[0])
    return scored_lines[0][1], scored_lines[1][1]

def perpendicular_distance_between_parallel_lines(line1, line2):
    _, _, C1 = line1
    _, _, C2 = line2
    return abs(C2 - C1)

def match_line(target_line, candidate_lines, tol_angle=30, tol_offset=60):
    x1_t, y1_t, x2_t, y2_t = target_line
    # Compute target angle in radians
    target_angle = np.degrees(np.arctan2(y2_t - y1_t, x2_t - x1_t))
    best_candidate = None
    best_score = float("inf")
    for pts in candidate_lines:
        x1, y1, x2, y2 = pts
        cand_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # difference in orientation
        d_angle = abs(target_angle - cand_angle)
        # distance from one endpoint to the target line (for position matching)
        target_eq = line_to_abd(x1_t, y1_t, x2_t, y2_t)
        d_offset = min(point_line_distance(target_eq, (x1, y1)),
                       point_line_distance(target_eq, (x2, y2)))
        # A simple score that favors small angular differences and small offset differences
        score = d_angle + d_offset
        if score < best_score and d_angle < tol_angle and d_offset < tol_offset:
            best_score = score
            best_candidate = pts
    if best_candidate is None:
        # If no good candidate was found, return the original target line as fallback.
        return target_line
    return best_candidate

def transform_point(current_point, scale, translation):
    x_curr, y_curr = current_point
    t_x, t_y = translation

    x_init = scale * x_curr + t_x
    y_init = scale * y_curr + t_y

    return (x_init, y_init)

class player_tracker:

    def track_players(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        yolo = player_detection()
        positions = []
        counter = 0
        on_first_frame = True
        first_frame = None
        static_first_frame = None

        while True:
            ret, frame = cap.read()
            if on_first_frame:
                static_first_frame = frame.copy()
                first_frame = frame.copy()
            if not ret:
                break
            

            lines, _ = find_lines(frame)
            new_positions = yolo.find_players(frame)
            
            if on_first_frame:
                h, w = first_frame.shape[:2]
                image_center = (w/2, h/2)
                line1_pts, line2_pts = find_two_lines_closest_to_center(lines, image_center)

                # Convert these to line-equation form: A*x + B*y + C = 0
                line1_eq = line_to_abd(*line1_pts)
                line2_eq = line_to_abd(*line2_pts)

                init_distance = perpendicular_distance_between_parallel_lines(line1_eq, line2_eq)
                x_int1_init = compute_x_intercept(line1_eq)
                x_int2_init = compute_x_intercept(line2_eq)
                init_x_intercept = (x_int1_init + x_int2_init) / 2

                # For later matching in every frame, store the original endpoints.
                ref_line1_pts = line1_pts.copy()
                ref_line2_pts = line2_pts.copy()
                on_first_frame = False
                
            # Match the two reference lines
            curr_line1_pts = match_line(ref_line1_pts, lines)
            curr_line2_pts = match_line(ref_line2_pts, lines)
            ref_line1_pts = curr_line1_pts.copy()
            ref_line2_pts = curr_line2_pts.copy()
            
            curr_line1_eq = line_to_abd(*curr_line1_pts)
            curr_line2_eq = line_to_abd(*curr_line2_pts)
            
            curr_distance = perpendicular_distance_between_parallel_lines(curr_line1_eq, curr_line2_eq)
            
            if curr_distance != 0:
                scale_factor = init_distance / curr_distance
            else:
                scale_factor = 1.0

            x_int1_curr = compute_x_intercept(curr_line1_eq)
            x_int2_curr = compute_x_intercept(curr_line2_eq)
            curr_x_intercept = (x_int1_curr + x_int2_curr) / 2

            translation_x = init_x_intercept - scale_factor * curr_x_intercept
            
            scaled_new_positions = []
            for i in new_positions:
                scaled_new_positions.append(transform_point(i, scale_factor, (translation_x, 0)))
            for i in scaled_new_positions:
                positions.append((i[0], i[1], counter))
            counter += 1
            
        for i in positions:
            cv2.circle(first_frame, (int(i[0]), int(i[1])), 5, (0 + int(255 * i[2] / total_frames), 0 + int(255 * i[2] / total_frames), 255), -1)
        return static_first_frame, first_frame
    
    def save_player_tracking(self, video_path, output_path):
        static_frame, frame = self.track_players(video_path)
        cv2.imwrite(output_path[0:-4] + "_first_frame" + output_path[-4:], static_frame)
        cv2.imwrite(output_path, frame)
