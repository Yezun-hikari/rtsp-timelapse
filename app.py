import os
import cv2
import threading
import time
import json
import logging
from flask import Flask, Response, jsonify, render_template

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# --- Configuration ---
RTSP_URL = os.environ.get("RTSP_URL", "")
RTSP_USER = os.environ.get("RTSP_USER", "")
RTSP_PASSWORD = os.environ.get("RTSP_PASSWORD", "")

import urllib.parse
if RTSP_URL and RTSP_USER and RTSP_PASSWORD:
    # Safely insert credentials into the RTSP URL
    parsed = urllib.parse.urlsplit(RTSP_URL)
    encoded_user = urllib.parse.quote(RTSP_USER, safe="")
    encoded_pass = urllib.parse.quote(RTSP_PASSWORD, safe="")

    # If URL already has credentials, replace them. Otherwise insert.
    netloc = f"{encoded_user}:{encoded_pass}@{parsed.hostname}"
    if parsed.port:
        netloc += f":{parsed.port}"

    RTSP_URL = urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))

DATA_DIR = "data"
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
STATE_DIR = os.path.join(DATA_DIR, "state")
STATE_FILE = os.path.join(STATE_DIR, "state.json")

# Ensure directories exist
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

class Camera:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

        # Optimize OpenCV for RTSP (use TCP)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        # We explicitly use CAP_FFMPEG to prevent OpenCV from misinterpreting
        # URL-encoded characters (like %21) as an image sequence.
        while self.running:
            if self.rtsp_url:
                if self.cap is None or not self.cap.isOpened():
                    logging.info(f"Connecting to RTSP stream...")
                    self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.latest_frame = frame
                else:
                    logging.warning("Failed to read frame, reconnecting in 2s...")
                    if self.cap:
                        self.cap.release()
                    self.cap = None
                    time.sleep(2)
            else:
                logging.warning("No RTSP_URL provided.")
                time.sleep(5)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

class TimelapseManager:
    def __init__(self, camera):
        self.camera = camera
        self.active = False
        self.capture_interval = 1.0 # seconds
        self.load_state()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)
                    self.active = state.get("active", False)
                    logging.info(f"Loaded state: Timelapse active = {self.active}")
            except Exception as e:
                logging.error(f"Error loading state: {e}")

    def save_state(self):
        try:
            with open(STATE_FILE, "w") as f:
                json.dump({"active": self.active}, f)
        except Exception as e:
            logging.error(f"Error saving state: {e}")

    def start(self):
        self.active = True
        self.save_state()
        logging.info("Timelapse started.")

    def stop(self):
        self.active = False
        self.save_state()
        logging.info("Timelapse stopped. Starting rendering process...")
        threading.Thread(target=self.render_video, daemon=True).start()

    def render_video(self):
        logging.info("Rendering video...")
        frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
        if not frames:
            logging.warning("No frames to render.")
            return

        sharp_frames = []
        for frame_file in frames:
            filepath = os.path.join(FRAMES_DIR, frame_file)
            img = cv2.imread(filepath)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharp_frames.append((fm, filepath))

        if not sharp_frames:
             logging.warning("No valid frames read.")
             return

        # Sort by blurriness (variance of Laplacian), keep top 80% sharpest frames
        # Actually, since it's a continuous motion, throwing away frames might make the video jumpy.
        # But per user request, we filter out blurry frames.
        # Let's dynamically find a threshold or just keep frames above a certain percentile.
        # A simple approach: keep top 90% or a fixed threshold if it's very blurry.
        # For a 3D printer, we just sort them by time, but exclude ones with very low FM.
        # Let's calculate mean and std dev of FM to find outliers.
        fms = [f[0] for f in sharp_frames]
        mean_fm = sum(fms) / len(fms)

        # Keep frames that are at least 50% of the mean FM
        threshold = mean_fm * 0.5

        valid_filepaths = [f[1] for f in sharp_frames if f[0] >= threshold]

        # Sort valid filepaths chronologically again (they were chronological by filename)
        valid_filepaths.sort()

        if not valid_filepaths:
            logging.warning("No sharp frames found.")
            return

        first_img = cv2.imread(valid_filepaths[0])
        height, width, layers = first_img.shape

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        video_name = os.path.join(VIDEOS_DIR, f"timelapse_{timestamp_str}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))

        logging.info(f"Writing {len(valid_filepaths)} frames to {video_name}...")
        for image_file in valid_filepaths:
            img = cv2.imread(image_file)
            video.write(img)

        cv2.destroyAllWindows()
        video.release()
        logging.info(f"Video saved as {video_name}")

        # Cleanup frames
        for frame_file in frames:
            try:
                os.remove(os.path.join(FRAMES_DIR, frame_file))
            except Exception as e:
                logging.error(f"Failed to remove {frame_file}: {e}")

    def _capture_loop(self):
        while True:
            if self.active:
                frame = self.camera.get_frame()
                if frame is not None:
                    timestamp = time.time()
                    filepath = os.path.join(FRAMES_DIR, f"{timestamp:.3f}.jpg")
                    cv2.imwrite(filepath, frame)
                    logging.debug(f"Saved frame: {filepath}")
            time.sleep(self.capture_interval)

camera = Camera(RTSP_URL)
timelapse = TimelapseManager(camera)

def gen_frames():
    while True:
        frame = camera.get_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1) # Limit stream to ~10 fps for preview

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    return jsonify({"active": timelapse.active})

@app.route('/api/start', methods=['POST'])
def start_timelapse():
    timelapse.start()
    return jsonify({"status": "started"})

@app.route('/api/stop', methods=['POST'])
def stop_timelapse():
    timelapse.stop()
    return jsonify({"status": "stopped", "message": "Rendering started in background."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
