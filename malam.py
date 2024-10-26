import eventlet
eventlet.monkey_patch()
import socketio
import cv2
import base64
import time
import os
from ultralytics import YOLO
from datetime import datetime
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
from typing import Optional

# Global lock untuk menghindari race condition
file_lock = threading.Lock()

# Inisialisasi Flask dan Socket.IO
sio = socketio.Server(cors_allowed_origins='*')
# sio.instrument(auth={

app = Flask(__name__)
CORS(app)
flask_app = socketio.WSGIApp(sio, app)

DATA_FILE = 'data.json'

# Fungsi untuk encode gambar ke base64
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Fungsi untuk menyimpan gambar dengan confidence tertinggi
def save_best_image(frame, confidence, folder, filename_prefix):
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{filename_prefix}_{timestamp}_conf-{confidence:.2f}.jpg"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame)
    print(f"Image saved at: {filepath}")

# Fungsi untuk mendapatkan file gambar terbaru
def get_latest_file(folder: str) -> Optional[str]:
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
        return max(files, key=os.path.getmtime) if files else None
    except Exception as e:
        print(f"Error finding latest file: {e}")
        return None


def detect_from_camera(model):
    up_folder = './up'
    down_folder = './down'
    os.makedirs(up_folder, exist_ok=True)
    os.makedirs(down_folder, exist_ok=True)

    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(3)
    cam1_available = cap1.isOpened()
    cam2_available = cap2.isOpened()

    if not cam1_available and not cam2_available:
        print("No cameras available.")
        return

    last_emit_time = time.time()
    try:
        while True:
            frame1, frame2 = None, None
            if cam1_available:
                ret1, frame1 = cap1.read()
                if not ret1:
                    print("Camera 1 disconnected.")
                    cam1_available = False

            if cam2_available:
                ret2, frame2 = cap2.read()
                if not ret2:
                    print("Camera 2 disconnected.")
                    cam2_available = False

            if not cam1_available and not cam2_available:
                print("Both cameras are unavailable. Exiting.")
                break

            if frame1 is not None:
                results = model(frame1)
                box_detection = [det for det in results[0].boxes if det.cls[0] == 0]
                for det in box_detection:
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if box_detection:
                    best_detection = max(box_detection, key=lambda det: (det.conf[0], (det.xyxy[0][2] - det.xyxy[0][0]) * (det.xyxy[0][3] - det.xyxy[0][1])))
                    confidence = best_detection.conf[0]
                    if frame1 is not None and frame1.size != 0:
                        save_best_image(frame1, confidence, up_folder, 'up_image')
                    if frame2 is not None and frame2.size != 0:
                        save_best_image(frame2, confidence, down_folder, 'down_image')
                time.sleep(0.1)

            burst_image_up = encode_image_to_base64(frame1) if frame1 is not None else None
            burst_image_down = encode_image_to_base64(frame2) if frame2 is not None else None

            if time.time() - last_emit_time >= 1:
                result = {
                    'BurstImage': {
                        'device_up': {'Image': burst_image_up} if burst_image_up else "No data",
                        'device_down': {'Image': burst_image_down} if burst_image_down else "No data"
                    }
                }
                try:
                    sio.emit('ResultDetection', result)
                    last_emit_time = time.time()
                except Exception as e:
                    print(f"Error emitting result: {e}")
    finally:
        if cam1_available:
            cap1.release()
        if cam2_available:
            cap2.release()
        cv2.destroyAllWindows()

def emit_saved_images():
    while True:
        up_image_path = get_latest_file('./up')
        if up_image_path:
            image_64 = encode_image_to_base64(cv2.imread(up_image_path))
            sio.emit('up', {'data': image_64})

        down_image_path = get_latest_file('./down')
        if down_image_path:
            image_64 = encode_image_to_base64(cv2.imread(down_image_path))
            sio.emit('down', {'data': image_64})

        time.sleep(1)

@sio.event
def connect(sid, environ):
    print(f'Client connected: {sid}')

if __name__ == '__main__':
    model = YOLO('bola ijo.pt')
    app.debug = True
    
    # Memulai task latar belakang untuk deteksi kamera dan pengiriman gambar
    sio.start_background_task(detect_from_camera, model=model)
    sio.start_background_task(emit_saved_images)

    # Menggunakan Eventlet sebagai server untuk menggabungkan Flask dan Socket.IO
    eventlet.wsgi.server(eventlet.listen(('', 5000)), flask_app)
