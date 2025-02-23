import cv2
import sys
import numpy as np
from line_detection import find_lines
from yolo10_detection import player_detection

def line_to_abd(x1, y1, x2, y2):
    """
    Convert endpoints to a normalized line equation (A*x + B*y + C = 0)
    so that the (A, B) vector has unit norm.
    """
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    norm = np.sqrt(A**2 + B**2) + 1e-8  # Avoid division by zero
    return (A / norm, B / norm, C / norm)

def compute_x_intercept(line_eq):
    """
    For a given line (A, B, C) representing A*x + B*y + C = 0,
    compute the x-intercept as if the line were extended infinitely.
    This is defined as the location where y=0.
    """
    A, B, C = line_eq
    if abs(A) < 1e-8:
        # For horizontal lines, the x-intercept is not defined.
        return None
    return -C / A

def point_line_distance(line, point):
    """
    Compute perpendicular distance from a point to a line given in form (A, B, C)
    """
    A, B, C = line
    x0, y0 = point
    return abs(A*x0 + B*y0 + C) / np.sqrt(A**2 + B**2 + 1e-8)

def find_two_lines_closest_to_center(lines, center):
    """
    Given a list of line endpoints, find the two lines whose distance (using the
    point-to-line distance) from the image center is smallest.
    Returns a tuple [(line1 endpoints), (line2 endpoints)].
    """
    scored_lines = []
    for pts in lines:
        x1, y1, x2, y2 = pts
        vec_line = line_to_abd(x1, y1, x2, y2)
        d = point_line_distance(vec_line, center)
        scored_lines.append((d, pts))
    scored_lines.sort(key=lambda x: x[0])
    if len(scored_lines) < 2:
        raise ValueError("Not enough lines detected.")
    # Optionally ensure the two lines are not nearly identical (e.g. check difference in orientation)
    return scored_lines[0][1], scored_lines[1][1]

def perpendicular_distance_between_parallel_lines(line1, line2):
    """
    For two normalized line equations (A*x+B*y+C=0 where sqrt(A^2+B^2)=1),
    the distance is simply the absolute difference in C.
    """
    _, _, C1 = line1
    _, _, C2 = line2
    return abs(C2 - C1)

def match_line(target_line, candidate_lines, tol_angle=30, tol_offset=60):
    """
    Given a target line (as endpoints) and a list of candidate lines (endpoints),
    find the candidate that most closely matches the target in terms of angle and offset.
    Returns the matched candidate endpoints.
    """
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
    """
    Maps a point from the current frame to the coordinates of the initial frame
    using the given scaling factor and translational offset.
    
    Parameters:
      current_point (tuple of float): The (x, y) coordinate in the current frame.
      scale (float): The scaling factor, e.g. s = init_distance / curr_distance.
      translation (tuple of float): The translation vector (t_x, t_y) that adjusts for panning.
    
    Returns:
      tuple of float: The (x, y) coordinate in the initial frame.
    
    The transformation applies:
      x_initial = scale * x_current + t_x
      y_initial = scale * y_current + t_y
    """
    # Unpack current point and translation components
    x_curr, y_curr = current_point
    t_x, t_y = translation

    # Compute the transformed coordinates
    x_init = scale * x_curr + t_x
    y_init = scale * y_curr + t_y

    return (x_init, y_init)

# Load your image
cap = cv2.VideoCapture("Play 1.mp4")
yolo = player_detection()
positions = []
counter = 0
on_first_frame = True
first_frame = None

# Check if the video was opened successfully
if not cap.isOpened():
    print("Could not open the video.")
    exit()

while True:
    ret, frame = cap.read()
    if on_first_frame:
        first_frame = frame.copy()
    if not ret:
        break
    

    lines, angles = find_lines(frame)
    new_positions = yolo.find_players(frame)
    
    if on_first_frame:
        h, w = first_frame.shape[:2]
        image_center = (w/2, h/2)
        # Find two lines that are closest to the center
        line1_pts, line2_pts = find_two_lines_closest_to_center(lines, image_center)

        # Convert these to line-equation form: A*x + B*y + C = 0
        line1_eq = line_to_abd(*line1_pts)
        line2_eq = line_to_abd(*line2_pts)

        # Compute the initial distance between the two lines
        init_distance = perpendicular_distance_between_parallel_lines(line1_eq, line2_eq)
        # Compute the initial x-intercepts for each line (if one returns None, consider an alternative approach)
        x_int1_init = compute_x_intercept(line1_eq)
        x_int2_init = compute_x_intercept(line2_eq)
        if x_int1_init is None or x_int2_init is None:
            raise ValueError("One of the initial lines is horizontal; consider using y-intercept instead.")
        init_x_intercept = (x_int1_init + x_int2_init) / 2

        # For later matching in every frame, store the original endpoints.
        ref_line1_pts = line1_pts.copy()
        ref_line2_pts = line2_pts.copy()
        on_first_frame = False
        
    # --- Match the two reference lines ---
    curr_line1_pts = match_line(ref_line1_pts, lines)
    curr_line2_pts = match_line(ref_line2_pts, lines)
    ref_line1_pts = curr_line1_pts.copy()
    ref_line2_pts = curr_line2_pts.copy()
    
    # Convert to line eq form
    curr_line1_eq = line_to_abd(*curr_line1_pts)
    curr_line2_eq = line_to_abd(*curr_line2_pts)
    
    # Compute the current perpendicular distance between the two lines
    curr_distance = perpendicular_distance_between_parallel_lines(curr_line1_eq, curr_line2_eq)
    
    # Compute scale factor. If the spacing has shrunk, assume zooming out.
    if curr_distance != 0:
        scale_factor = init_distance / curr_distance
    else:
        scale_factor = 1.0

     # Compute the current x-intercepts for each line and average them
    x_int1_curr = compute_x_intercept(curr_line1_eq)
    x_int2_curr = compute_x_intercept(curr_line2_eq)
    if x_int1_curr is None or x_int2_curr is None:
        continue  # Skip frame if one of the lines is horizontal
    curr_x_intercept = (x_int1_curr + x_int2_curr) / 2

    # Calculate horizontal translation adjusted by the scaling factor:
    translation_x = init_x_intercept - scale_factor * curr_x_intercept
    
    scaled_new_positions = []
    for i in new_positions:
        scaled_new_positions.append(transform_point(i, scale_factor, (translation_x, 0)))

    # # Display the warped image
    # for line in lines:
    #     x1, y1, x2, y2 = line  # Extract points from the nested array
    #     cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.line(frame, (curr_line1_pts[0], curr_line1_pts[1]), (curr_line1_pts[2], curr_line1_pts[3]), (0, 255, 0), 2)
    cv2.line(frame, (curr_line2_pts[0], curr_line2_pts[1]), (curr_line2_pts[2], curr_line2_pts[3]), (255, 0, 0), 2)
    # Show current scale and translation on the image.
    msg = "Scale: {:.2f}  Trans: ({:.1f})".format(scale_factor, translation_x)
    cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for i in scaled_new_positions:
        positions.append((i[0], i[1], counter))
    for i in positions:
        cv2.circle(frame, (int(i[0]), int(i[1])), 5, (0 + i[2] * 2, 0 + i[2] * 2, 255), -1)
    counter += 1
    cv2.imshow("frame", frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
for i in positions:
    cv2.circle(first_frame, (int(i[0]), int(i[1])), 5, (0 + i[2] * 2, 0 + i[2] * 2, 255), -1)
    cv2.imshow("first frame", first_frame)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
sys.exit()
