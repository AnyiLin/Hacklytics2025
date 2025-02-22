#!/usr/bin/env python3
"""
Implementation of:
"Using Computer Vision and Machine Learning to Automatically Classify NFL Game Film and Develop a Player Tracking System"
(as presented in the 2018 Research Papers Competition)

This implementation follows the paper’s steps:
1. Standardizing game film by detecting field lines and rotating the shot.
2. Detecting offensive players by jersey color and the quarterback via a blue marker.
3. Generating coordinate features and running multiple ML classifiers (simulated) to identify formations.
4. Tracking player positions over time to compute route distances and speeds.
5. Merging formation data with play-by-play information for further analysis.

NOTE: This demonstration code uses simulated data in places (e.g. dummy formation labels)
because a complete NFL dataset is not publicly available. Adjust HSV thresholds, conversion factors,
and parameters as necessary.
"""

import cv2
import numpy as np
import glob
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

FRAME_RATE = 5  # frames per second (fps)

# ------------------------------------------------------------------------------
# Section 1. Preprocessing and Field Geometry Correction
# ------------------------------------------------------------------------------

def load_images(image_dir):
    """Load all images from the given directory."""
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    images = []
    for fname in image_files:
        img = cv2.imread(fname)
        if img is not None:
            images.append(img)
    return images

def detect_field_lines(image):
    """
    Detect full field white lines using Canny edge detection and Hough Transform.
    (Additional heuristics would normally be applied to isolate the full field lines.)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Use probabilistic Hough transform:
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=100, maxLineGap=10)
    # Filter for nearly horizontal lines (assumed to be the full field lines)
    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Compute angle in degrees
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 10 or abs(angle - 180) < 10:
                horizontal_lines.append((x1, y1, x2, y2))
    return horizontal_lines

def compute_rotation_angle(lines):
    """
    Compute the average angle of the detected horizontal lines.
    The paper describes calculating the shooter's angle via arccos(x). Here we simply use the average.
    """
    angles = []
    for (x1, y1, x2, y2) in lines:
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        angles.append(angle)
    if angles:
        avg_angle = np.mean(angles)
    else:
        avg_angle = 0
    # Return a negative angle to correct the rotation
    return -avg_angle

def rotate_image(image, angle):
    """Rotate image by the given angle (in degrees)."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# ------------------------------------------------------------------------------
# Section 2. Player Detection (Offensive players and Quarterback)
# ------------------------------------------------------------------------------

def detect_offensive_players(image):
    """
    Detect offensive players using jersey color segmentation.
    In this example we assume the offensive jersey is burgundy. Adjust the HSV thresholds as needed.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Approximate burgundy color range (change based on calibration)
    lower_burgundy = np.array([160, 50, 50])
    upper_burgundy = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_burgundy, upper_burgundy)
    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    players = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:  # filter out small noise contours
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            players.append((cx, cy))
    return players, mask

def detect_quarterback(image):
    """
    Detect the quarterback using a blue marker (assumed to be drawn as a blue square).
    Adjust the HSV thresholds for the blue as necessary.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    qb = None
    if contours:
        # Choose the largest blue contour as the quarterback marker
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        qb = (x + w // 2, y + h // 2)
    return qb, mask

def sort_players_by_x(players):
    """Sort players by x–coordinate (left-to-right) to impose an order on the feature vector."""
    return sorted(players, key=lambda p: p[0])

def extract_features(players, quarterback):
    """
    Create a feature vector from each player’s coordinates relative to the quarterback.
    For simplicity we conjoin the differential (x, y) into one list.
    """
    if quarterback is None:
        return None
    qb_x, qb_y = quarterback
    features = []
    for (px, py) in players:
        features.append(px - qb_x)
        features.append(py - qb_y)
    return features

# ------------------------------------------------------------------------------
# Section 3. Formation Identification using Machine Learning
# ------------------------------------------------------------------------------

def generate_dummy_dataset(n_samples=500, n_features=20, n_classes=29):
    """
    Simulate a training dataset of coordinate features extracted from ~500 labeled formation images.
    In practice you would build this from your manual data labeling process.
    """
    X = np.random.randint(-50, 50, size=(n_samples, n_features))
    y = np.random.randint(0, n_classes, size=(n_samples,))
    return X, y

def train_classifiers(X, y):
    """
    Train five classifiers on the training dataset and report their in-sample accuracies.
    (In the paper, Classification and Regression Trees—i.e. Decision Trees—worked best.)
    """
    classifiers = {
         "DecisionTree": DecisionTreeClassifier(random_state=42),
         "NaiveBayes": GaussianNB(),
         "SVM": SVC(gamma='auto', random_state=42),
         "KNeighbors": KNeighborsClassifier(),
         "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
    }
    results = {}
    print("Classifier Accuracy on Training Data:")
    for name, clf in classifiers.items():
        clf.fit(X, y)
        y_pred = clf.predict(X)
        accuracy = accuracy_score(y, y_pred) * 100.0
        results[name] = (clf, accuracy)
        print(f"  {name}: {accuracy:.1f}%")
    return results

def classify_formation(feature_vector, classifier):
    """Use the chosen classifier to predict a formation label from the feature vector."""
    if feature_vector is None:
        return None
    feature_vector = np.array(feature_vector).reshape(1, -1)
    formation = classifier.predict(feature_vector)
    return formation[0]

# ------------------------------------------------------------------------------
# Section 4. Player Tracking and Speed Calculation
# ------------------------------------------------------------------------------

def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two (x, y) points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def compute_conversion_factor(image):
    """
    Compute the pixel-to-yard conversion factor from the white field lines.
    In the paper, the distance between full-field white lines (repeated every 5 yards)
    is used to derive the conversion. Here we assume a dummy constant factor.
    """
    return 10.0  # dummy: assume 10 pixels per yard

def compute_player_speed(tracked_positions, pixels_per_yard, frame_rate=FRAME_RATE):
    """
    Given a list of tracked positions over time (frames), compute the per-interval speed
    in yards per second and return the average speed (converted to mph).
    
    Conversion: 1 yard/s ≈ 2.045 mph.
    """
    distances = []
    for i in range(1, len(tracked_positions)):
        d_px = euclidean_distance(tracked_positions[i-1], tracked_positions[i])
        d_yards = d_px / pixels_per_yard
        distances.append(d_yards)
    # Compute speed for each interval (each frame is 1/frame_rate seconds apart)
    speeds_yps = [d * frame_rate for d in distances]
    avg_speed_yps = sum(speeds_yps) / len(speeds_yps) if speeds_yps else 0
    avg_speed_mph = avg_speed_yps * 2.045
    return avg_speed_mph, speeds_yps

def simulate_tracking():
    """
    Simulate tracking for two players over three frames.
    For example, the paper compared DeSean Jackson vs. Maurice Harris.
    (Note: These coordinates are dummy values.)
    """
    # Simulated positions (pixels) for DeSean Jackson across successive frames
    jackson_positions = [(100, 200), (120, 205), (140, 210)]
    # Simulated positions for Maurice Harris
    harris_positions = [(100, 200), (110, 203), (120, 205)]
    pixels_per_yard = 10.0
    jackson_speed, jackson_speeds = compute_player_speed(jackson_positions, pixels_per_yard)
    harris_speed, harris_speeds = compute_player_speed(harris_positions, pixels_per_yard)
    print("\nPlayer Tracking and Speed Analysis:")
    print(f"DeSean Jackson average speed: {jackson_speed:.2f} mph (interval speeds: {jackson_speeds})")
    print(f"Maurice Harris average speed: {harris_speed:.2f} mph (interval speeds: {harris_speeds})")

# ------------------------------------------------------------------------------
# Section 5. Merging Formation with Play-by-play Data
# ------------------------------------------------------------------------------

def merge_play_by_play_and_formation():
    """
    Simulate merging the automatically determined formation with play-by-play game data.
    In practice the formation label can be joined with data (e.g. down, distance, time, etc.)
    to help uncover coaching tendencies.
    """
    data = {
       "Image": ["STLatWAS_1", "STLatWAS_2"],
       "Formation": ["Singleback Ace", "Spread Gun"],
       "Play-by-play": [
           "1ST & 10 AT WAS 19 (13:45) M.Jones right guard to WAS 17 for -2 yards (M.Brockers)",
           "(Shotgun) K.Cousins pass short left to P.Garçon to WAS 21 for 4 yards (J.Jenkins)"
       ]
    }
    df = pd.DataFrame(data)
    print("\n--- Combined Formation & Play-by-Play Data ---")
    print(df.to_string(index=False))

# ------------------------------------------------------------------------------
# Main pipeline: tie all pieces together
# ------------------------------------------------------------------------------

def main():
    ### STEP 1: Load and Preprocess a Sample Screenshot
    image_path = "frame.png"  # replace with your sample image path
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found! Please place a sample image in the working directory.")
        return

    # Detect field lines and compute rotation angle
    lines = detect_field_lines(img)
    rotation_angle = compute_rotation_angle(lines)
    rotated_img = rotate_image(img, rotation_angle)
    cv2.imwrite("rotated_image.jpg", rotated_img)
    print(f"Image rotated by {rotation_angle:.2f} degrees based on field line detection.")

    ### STEP 2: Detect Offensive Players and Locate the Quarterback
    players, burgundy_mask = detect_offensive_players(rotated_img)
    qb, blue_mask = detect_quarterback(rotated_img)
    if qb is None:
        print("Quarterback not detected – please verify the blue marker threshold!")
    else:
        print(f"Quarterback detected at {qb}.")

    # (For visualization) Mark all detected players on the image.
    for (x, y) in players:
        cv2.circle(rotated_img, (x, y), 5, (0, 255, 0), -1)
    if qb is not None:
        cv2.circle(rotated_img, qb, 7, (255, 0, 0), -1)
    cv2.imwrite("players_detected.jpg", rotated_img)
    print("Players and QB markers have been drawn on the image, and the result saved as 'players_detected.jpg'.")

    ### STEP 3: Formation Identification via ML
    sorted_players = sort_players_by_x(players)
    features = extract_features(sorted_players, qb)
    # For formation classification we require a fixed-length feature vector.
    n_features_required = 20
    if features is None:
        features = [0] * n_features_required
    else:
        if len(features) < n_features_required:
            features.extend([0] * (n_features_required - len(features)))
        elif len(features) > n_features_required:
            features = features[:n_features_required]

    # Load (simulate) a training dataset of formation labels.
    X_train, y_train = generate_dummy_dataset(n_samples=500,
                                              n_features=n_features_required,
                                              n_classes=29)
    classifier_results = train_classifiers(X_train, y_train)
    # Choose the best classifier – here DecisionTree (CART) per the paper’s results
    clf_cart, acc = classifier_results["DecisionTree"]
    predicted_formation = classify_formation(features, clf_cart)
    print(f"\nPredicted formation label (dummy label): {predicted_formation}")
    # In practice you would map this numeric label to a formation name (e.g. 'Singleback Ace', etc.)

    ### STEP 4: Player Tracking & Speed Computation
    simulate_tracking()

    ### STEP 5: Merge with Play-by-play Data
    merge_play_by_play_and_formation()

if __name__ == "__main__":
    main()