from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys

# Add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from player_tracker import player_tracker
from werkzeug.utils import secure_filename
import glob
from ai_service import get_play_analysis

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plays"
)
ALLOWED_EXTENSIONS = {"mp4"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_next_play_number():
    existing_files = glob.glob(os.path.join(app.config["UPLOAD_FOLDER"], "Play *.mp4"))
    if not existing_files:
        return 1
    numbers = [int(f.split("Play ")[1].split(".")[0]) for f in existing_files]
    return max(numbers) + 1


@app.route("/upload", methods=["POST"])
def upload_file():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Create directories if they don't exist
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        os.makedirs(
            os.path.join(os.path.dirname(app.config["UPLOAD_FOLDER"]), "images"),
            exist_ok=True,
        )

        play_number = get_next_play_number()
        filename = f"Play {play_number}.mp4"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Run player tracking with switched parameter order
        tracker = player_tracker()
        first_frame_path = f"images/play_{play_number}_map_first_frame.jpg"
        final_map_path = f"images/play_{play_number}_map.jpg"
        tracker.save_player_tracking_with_first_frame(
            filepath,
            os.path.join("..", first_frame_path),  # First frame
            os.path.join("..", final_map_path),  # Movement map
        )

        # Get AI analysis
        analysis = get_play_analysis(play_number)

        return jsonify(
            {
                "success": True,
                "play_number": play_number,
                "first_frame_path": f"/images/{os.path.basename(first_frame_path)}",  # Add /images/ prefix
                "final_map_path": f"/images/{os.path.basename(final_map_path)}",  # Add /images/ prefix
                "analysis": analysis,
            }
        )


# Update image serving to use absolute path
@app.route("/images/<path:filename>")
def serve_image(filename):
    image_directory = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images"
    )
    return send_from_directory(image_directory, filename)


if __name__ == "__main__":
    app.run(debug=True)
