import socketio
import eventlet
import cv2
import base64
import time
import torch
import numpy as np
from ultralytics import YOLO

# Load model PyTorch dari file TorchScript
model = YOLO('yolov8m.pt')

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

# Variabel untuk menyimpan akurasi tertinggi
highest_confidence = 0
best_frame = None

# Variabel untuk menghitung FPS
prev_time = 0
fps = 0

# Fungsi untuk encode gambar ke base64
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Event ketika client connect
@sio.event
def connect(sid, environ):
    print(f'Client connected: {sid}')

# Fungsi untuk deteksi objek dari frame kamera
def detect_from_camera():
    global highest_confidence, best_frame, prev_time, fps

    # Membuka kamera
    cap = cv2.VideoCapture(0)

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

        sio.emit('image',{'data' : image_64})

        # Tampilkan FPS di pojok kiri atas
        cv2.putText(result_image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Tampilkan frame yang sedang diproses (opsional)
        # cv2.imshow('YOLOv8 Detection', result_image)

        # Jika 'q' ditekan, keluar dari loop
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        time.sleep(1)

    cap.release()
    cv2.destroyAllWindows()

# Jalankan server
if __name__ == '__main__':
    # Jalankan server Socket.IO
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)

    # Jalankan deteksi dari kamera di thread terpisah
    eventlet.spawn(detect_from_camera)

