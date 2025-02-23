from player_tracker import player_tracker

tracker = player_tracker()
number = "4"
tracker.save_player_tracking(
    "plays\Play " + number + ".mp4", "images\play_" + number + "_map.jpg"
)
