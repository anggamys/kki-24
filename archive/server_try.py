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

# Fungsi untuk encode gambar ke base64
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Fungsi untuk menyimpan gambar dengan akurasi tertinggi
def save_best_image(image, confidence, folder_path, filename):
    file_path = os.path.join(folder_path, filename)
    cv2.imwrite(file_path, image)
    print(f"Image saved at {file_path} with confidence {confidence:.2f}")

# Fungsi untuk mendeteksi manusia dari frame kamera
def detect_from_camera(device, model):
    # Pastikan folder untuk menyimpan gambar ada
    up_folder = './up'
    down_folder = './down'
    os.makedirs(up_folder, exist_ok=True)
    os.makedirs(down_folder, exist_ok=True)

    # Inisialisasi akurasi tertinggi
    highest_confidence_up = 0
    highest_confidence_down = 0

    # Tentukan sumber video
    if device == 'up':
        source = 0  # Kamera internal
    elif device == 'down':
        source = 1  # Kamera eksternal atau RTSP
    else:
        print("No valid camera detected")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open camera: {device}")
        return

    prev_time = cv2.getTickCount()  # Variabel untuk menghitung FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture frame from {device}")
            break

        # Hitung FPS
        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_time - prev_time)
        prev_time = current_time

        # Deteksi objek manusia (class 0) menggunakan YOLOv8
        results = model(frame)

        # Filter hasil deteksi untuk hanya manusia (class 0)
        human_detections = [det for det in results[0].boxes if det.cls[0] == 0]

        # Tampilkan frame dengan deteksi (opsional)
        for det in human_detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow(f'{device} frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if human_detections:
            # Ambil deteksi dengan confidence tertinggi
            best_detection = max(human_detections, key=lambda det: det.conf[0])
            confidence = best_detection.conf[0]

            # Simpan gambar jika confidence tertinggi baru
            if device == "up" and confidence > highest_confidence_up:
                highest_confidence_up = confidence
                save_best_image(frame, confidence, up_folder, 'best_up.jpg')

            elif device == 'down' and confidence > highest_confidence_down:
                highest_confidence_down = confidence
                save_best_image(frame, confidence, down_folder, 'best_down.jpg')

        # Kirim gambar ke client
        burst_image = encode_image_to_base64(frame)


        result = {
            'ResultDetection': {
                device: {
                    'Image': burst_image
                }
            }
        }

        sio.emit('ResultDetection', result)

        time.sleep(0.06)  # Kontrol frekuensi deteksi

    cap.release()
    cv2.destroyAllWindows()

# Fungsi untuk mengirim gambar yang tersimpan ke client setiap 1 detik
def emit_saved_images():
    while True:
        # Emit gambar up
        up_image_path = './up/best_up.jpg'
        if os.path.exists(up_image_path):
            image_64 = encode_image_to_base64(cv2.imread(up_image_path))
            sio.emit('up', {'data': image_64})

        # Emit gambar down
        down_image_path = './down/best_down.jpg'
        if os.path.exists(down_image_path):
            image_64 = encode_image_to_base64(cv2.imread(down_image_path))
            sio.emit('down', {'data': image_64})

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
    # Load model YOLO sekali untuk digunakan di kedua kamera
    model = YOLO('yolov8m.pt')

    # Menjalankan deteksi untuk kamera internal dan eksternal
    sio.start_background_task(detect_from_camera, device='up', model=model)
    sio.start_background_task(detect_from_camera, device='down', model=model)

    # Memulai task untuk mengirim gambar yang tersimpan setiap 1 detik
    sio.start_background_task(emit_saved_images)

    # Jalankan server Socket.IO
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
