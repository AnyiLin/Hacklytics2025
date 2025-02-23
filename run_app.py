import subprocess
import time
import os
import sys
import signal


def run_servers():
    # Get the absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    server_dir = os.path.join(base_dir, "server")
    client_dir = os.path.join(base_dir, "client")

    # Start Flask server
    print("Starting Flask server...")
    flask_process = subprocess.Popen(["python3", "app.py"], cwd=server_dir)
    time.sleep(2)  # Give Flask time to start

    # Start Svelte dev server
    print("Starting Svelte development server...")
    npm_process = subprocess.Popen(["pnpm", "run", "dev"], cwd=client_dir)

    def cleanup(signum, frame):
        print("\nShutting down servers...")
        flask_process.terminate()
        npm_process.terminate()
        sys.exit(0)

    # Register the signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("\nServers are running!")
    print("Access the application at: http://localhost:3000")
    print("API server is running at: http://localhost:5000")
    print("Press Ctrl+C to stop both servers\n")

    try:
        # Keep the script running
        flask_process.wait()
        npm_process.wait()
    except KeyboardInterrupt:
        cleanup(None, None)


if __name__ == "__main__":
    run_servers()
