import eventlet
eventlet.monkey_patch()
import socketio
import cv2
import base64
import time
import os
import torch
from ultralytics import YOLO
import requests

# Fungsi untuk memulai GoPro
def start_gopro():
    url = 'http://172.25.165.51:8080/gopro/webcam/start?res=4&fov=4&port=8556&protocol=RTSP'
    response = requests.get(url)
    if response.status_code == 200:
        print("GoPro started in webcam mode")
    else:
        print(f"Failed to start GoPro: {response.text}")

# Fungsi untuk encode gambar ke base64
def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
    return base64.b64encode(image_data).decode('utf-8')

# Fungsi untuk menyimpan gambar dengan akurasi tertinggi
def save_best_image(image, confidence, folder_path, filename):
    file_path = os.path.join(folder_path, filename)
    cv2.imwrite(file_path, image)
    print(f"Image saved at {file_path} with confidence {confidence:.2f}")

# Fungsi untuk mendeteksi manusia dari frame kamera
def detect_from_camera(device):
    # Load model YOLOv8 dan konfigurasi deteksi hanya manusia (class 0 adalah manusia)
    model = YOLO('yolov8m.pt')
    
    # Pastikan folder untuk menyimpan gambar ada
    up_folder = './up'
    down_folder = './down'
    os.makedirs(up_folder, exist_ok=True)
    os.makedirs(down_folder, exist_ok=True)

    # Inisialisasi akurasi tertinggi
    highest_confidence_up = 0
    highest_confidence_down = 0

    # Tentukan sumber video
    if device == 'cam1':
        source = 0  # Kamera internal
    elif device == 'gopro':
        source = "rtsp://172.25.165.51:554/live"  # GoPro
    else:
        print("No valid camera detected")
        return

    cap = cv2.VideoCapture(source)
    prev_time = 0  # Variabel untuk menghitung FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Hitung FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        # Deteksi objek manusia (class 0) menggunakan YOLOv8
        results = model(frame)

        # Filter hasil deteksi untuk hanya manusia (class 0)
        human_detections = [det for det in results[0].boxes if det.cls[0] == 0]

        if human_detections:
            # Ambil deteksi dengan confidence tertinggi
            best_detection = max(human_detections, key=lambda det: det.conf[0])
            confidence = best_detection.conf[0]

            # Simpan gambar jika confidence tertinggi baru
            if device == "cam1" and confidence > highest_confidence_up:
                highest_confidence_up = confidence
                save_best_image(frame, confidence, up_folder, 'best_up.jpg')

            elif device == 'gopro' and confidence > highest_confidence_down:
                highest_confidence_down = confidence
                save_best_image(frame, confidence, down_folder, 'best_down.jpg')

        # Delay untuk mengontrol frekuensi deteksi
        time.sleep(0.06)

    cap.release()
    cv2.destroyAllWindows()

# Fungsi untuk mengirim gambar yang tersimpan ke client setiap 1 detik
def emit_saved_images():
    while True:
        # Emit gambar up
        up_image_path = './up/best_up.jpg'
        if os.path.exists(up_image_path):
            image_64 = encode_image_to_base64(up_image_path)
            sio.emit('up', {'data': image_64})

        # Emit gambar down
        down_image_path = './down/best_down.jpg'
        if os.path.exists(down_image_path):
            image_64 = encode_image_to_base64(down_image_path)
            sio.emit('down', {'data': image_64})

        # Tunggu 1 detik sebelum emit berikutnya
        time.sleep(1)

# Inisialisasi Socket.IO server
sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio)

# Event ketika client connect
@sio.event
def connect(sid, environ):
    print(f'Client connected: {sid}')

# Jalankan server
if __name__ == '__main__':
    start_gopro()  # Start GoPro

    # Menjalankan deteksi untuk GoPro dan kamera internal
    sio.start_background_task(detect_from_camera, device='gopro')
    sio.start_background_task(detect_from_camera, device='cam1')

    # Memulai task untuk mengirim gambar yang tersimpan setiap 1 detik
    sio.start_background_task(emit_saved_images)

    # Jalankan server Socket.IO
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
