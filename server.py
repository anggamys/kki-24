import eventlet
eventlet.monkey_patch()
import socketio
import cv2
import base64
import time
import torch
import numpy as np
from ultralytics import YOLO
import requests



def start_gopro():
    url = 'http://172.25.165.51:8080/gopro/webcam/start?res=12&fov=0&port=8556&protocol=RTSP'

    # Request untuk memulai GoPro dalam mode webcam
    response = requests.request("GET", url)

    if response.status_code == 200:
        print("GoPro started in webcam mode")
    else:
        print(f"Failed to start GoPro: {response.text}")


# Fungsi untuk inferensi menggunakan PyTorch
def infer_pytorch(model, image):
    # Preprocess image: Ubah dari BGR ke RGB dan normalisasi
    image_tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konversi BGR ke RGB
    image_tensor = image_tensor / 255.0  # Normalisasi ke rentang [0, 1]
    image_tensor = torch.from_numpy(image_tensor).float().to("cuda")  # Ubah menjadi tensor dan kirim ke GPU
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Ubah bentuk ke (1, C, H, W)

    with torch.no_grad():
        output = model(image_tensor)  # Inferensi

    # Post-process output jika diperlukan
    # Misalnya: ambil bounding boxes, kelas, dan skor dari output
    return output.cpu().numpy()  # Kembalikan output ke CPU

# Inisialisasi Socket.IO server
sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio)

# Fungsi untuk encode gambar ke base64
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Event ketika client connect
@sio.event
def connect(sid, environ):
    print(f'Client connected: {sid}')

# Fungsi untuk deteksi objek dari frame kamera
def detect_from_camera(device):
    # Load model PyTorch dari file TorchScript
    model = YOLO('yolov8m.pt')
    # Variabel untuk menyimpan akurasi tertinggi
    highest_confidence = 0
    best_frame = None

    # Variabel untuk menghitung FPS
    prev_time = 0
    fps = 0

    if device == 'cam1':
        source = 0
    elif device == 'gopro':
        source = "rtsp://172.25.165.51:554/live"
    else:
        print("No device camera detected")

    # Membuka kamera
    cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Hitung FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Deteksi menggunakan PyTorch
        # output = infer_pytorch(model, frame)
        image = './bus.jpg'
        output = model(frame)

        result_image = output[0].plot()

        image_64 = encode_image_to_base64(result_image)

        # Emit gambar ke client
        if device == "cam1":
            sio.emit('up', {'data': image_64})
        elif device == 'gopro':
            sio.emit('down', {'data': image_64})

        # Tampilkan FPS di pojok kiri atas
        cv2.putText(result_image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Tampilkan frame yang sedang diproses (opsional)
        cv2.imshow('YOLOv8 Detection', result_image)

        # Jika 'q' ditekan, keluar dari loop
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        print(image_64[:10])
        time.sleep(0.06)

    cap.release()
    cv2.destroyAllWindows()

# Jalankan server
if __name__ == '__main__':
    # Start gopro
    start_gopro()

    # Running detection
    sio.start_background_task(detect_from_camera(device='gopro'))
    # sio.start_background_task(detect_from_camera(device='cam1'))
    
    # Jalankan server Socket.IO
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)

