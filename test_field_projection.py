import cv2
import sys
import numpy as np
from line_detection import find_lines
from yolo10_detection import player_detection
def line_to_abd(x1, y1, x2, y2):
    """
    Convert two endpoints into a line in the form A*x + B*y + C = 0.
    It returns (A, B, C). Note: (A, B) is not necessarily normalized.
    """
    A = y2 - y1
    B = x1 - x2
    C = x2*y1 - x1*y2
    return (A, B, C)

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
    Compute the perpendicular distance between two (approximately parallel) lines.
    Assumes that line1 and line2 are given as (A, B, C) in similar scale.
    """
    A, B, C1 = line1
    _, _, C2 = line2
    norm_factor = np.sqrt(A**2 + B**2)
    return abs(C2 - C1) / (norm_factor + 1e-8)

def compute_midpoint_of_lines(line1, line2, ref_point):
    """
    One simple approach to define a 'location' for a line is to compute the point on the
    line that is closest to a reference point (for example, the image center).
    Then, we can take the average of those two points as a midpoint.
    """
    def closest_point(line, pt):
        A, B, C = line
        x0, y0 = pt
        # Compute the projection of pt onto the line:
        t = (A*x0 + B*y0 + C) / (A**2+B**2+1e-8)
        # The projected point (x', y') that lies on the line (using normal!)
        x_proj = x0 - A*t
        y_proj = y0 - B*t
        return np.array([x_proj, y_proj])
    p1 = closest_point(line1, ref_point)
    p2 = closest_point(line2, ref_point)
    return (p1 + p2) / 2

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
        first_frame = frame
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
        # Also compute a representative midpoint (using projection onto each line) for later panning measurement.
        init_midpoint = compute_midpoint_of_lines(line1_eq, line2_eq, image_center)

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

    # Compute current midpoint (for panning)
    curr_midpoint = compute_midpoint_of_lines(curr_line1_eq, curr_line2_eq, image_center)
    # Translation (in pixel units) is the shift from current midpoint (after unscaling) to init_midpoint.
    # (Depending on whether you want to map current to initial, you might perform: p_initial = scale*(p_current) + t)
    translation = np.array(init_midpoint) - scale_factor * np.array(curr_midpoint)
    # For visualization, draw the detected/matched lines
    vis = frame.copy()
    cv2.line(vis, (curr_line1_pts[0], curr_line1_pts[1]), (curr_line1_pts[2], curr_line1_pts[3]), (0, 255, 0), 2)
    cv2.line(vis, (curr_line2_pts[0], curr_line2_pts[1]), (curr_line2_pts[2], curr_line2_pts[3]), (255, 0, 0), 2)
    
    # Draw the midpoint
    mid_pt_int = tuple(np.int32(curr_midpoint))
    cv2.circle(vis, mid_pt_int, 5, (0, 0, 255), -1)
    
    # Show current scale and translation on the image.
    msg = "Scale: {:.2f}  Trans: ({:.1f}, {:.1f})".format(scale_factor, translation[0], translation[1])
    cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.imshow("Tracked Lines", vis)
    
    
    # Display the warped image
    for line in lines:
        x1, y1, x2, y2 = line  # Extract points from the nested array
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    for i in new_positions:
        positions.append((i[0], i[1], counter))
    for i in positions:
        cv2.circle(frame, (int(i[0]), int(i[1])), 5, (0 + i[2] * 2, 0 + i[2] * 2, 255), -1)
    counter += 1
    cv2.imshow("frame", frame)
    
    cv2.waitKey(0)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()
cap.release()
cv2.destroyAllWindows()
