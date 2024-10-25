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

# Kunci global untuk menghindari race condition saat mengakses file
file_lock = threading.Lock()

# Inisialisasi Flask dan Socket.IO
sio = socketio.Server(cors_allowed_origins='*')
app = Flask(__name__)
CORS(app)  # Mengizinkan CORS untuk aplikasi Flask
flask_app = socketio.WSGIApp(sio, app)  # Gabungkan Flask dengan WebSocket

DATA_FILE = 'data.json'

# Fungsi untuk encode gambar ke base64
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Fungsi untuk menyimpan gambar dengan akurasi tertinggi
def save_best_image(frame, confidence, folder, filename_prefix):
    # Buat folder jika belum ada
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Format nama file menggunakan timestamp dan confidence
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{filename_prefix}_{timestamp}_conf-{confidence:.2f}.jpg"
    filepath = os.path.join(folder, filename)

    # Simpan gambar
    cv2.imwrite(filepath, frame)
    print(f"Image saved at: {filepath}")

# Fungsi untuk mencari file terbaru berdasarkan waktu modifikasi
def get_latest_file(folder: str) -> Optional[str]:
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
        latest_file = max(files, key=os.path.getmtime)
        return latest_file
    except ValueError:
        # Jika tidak ada file di folder, return None
        return None

# Fungsi untuk memuat data dari file JSON
def load_data():
    # Menggunakan kunci saat membuka file
    with file_lock:
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, 'r') as f:
                    # Menghindari masalah jika file kosong
                    file_content = f.read().strip()
                    if not file_content:
                        return {}
                    return json.loads(file_content)
            except json.JSONDecodeError:
                # Jika file rusak, kembalikan data kosong
                print("Error: JSON format is invalid, returning empty data.")
                return {}
        return {}

# Fungsi untuk menyimpan data ke file JSON
def save_data(data):
    # Menggunakan kunci saat menulis ke file
    with file_lock:
        try:
            with open(DATA_FILE, 'w') as f:
                json.dump(data, f, indent=4)
        except IOError as e:
            # Jika ada masalah saat menulis file
            print(f"Error saving data to {DATA_FILE}: {str(e)}")

# Fungsi untuk validasi struktur data
def validate_ball_data(data):
    required_keys = ['idBall', 'lat', 'long', 'typeBall', 'typeArena']
    errors = []  # Daftar untuk menyimpan pesan error

    # Cek apakah semua kunci yang diperlukan ada
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required field: '{key}'")

    # Validasi idBall (bisa string atau int)
    if not isinstance(data.get('idBall'), (int, str)):
        errors.append("Invalid 'idBall': must be an integer or string.")

    # Validasi lat dan long (keduanya harus angka)
    try:
        lat = float(data['lat'])
        long = float(data['long'])
    except (ValueError, KeyError):
        errors.append("Invalid coordinates: 'lat' and 'long' must be numbers.")

    # Validasi typeBall (hanya merah atau hijau)
    if data.get('typeBall') not in ['merah', 'hijau']:
        errors.append("Invalid 'typeBall': must be 'merah' or 'hijau'.")

    # Validasi typeArena (hanya a atau b)
    if data.get('typeArena') not in ['a', 'b']:
        errors.append("Invalid 'typeArena': must be 'a' or 'b'.")

    # Jika ada error, kembalikan False dan daftar error
    if errors:
        return False, {"errors": errors}

    return True, ""

@app.route('/api/ball', methods=['POST'])
def create_ball():
    try:
        new_ball = request.json

        # Validasi data bola
        is_valid, validation_response = validate_ball_data(new_ball)
        if not is_valid:
            return jsonify({'success': False, 'message': 'Validation errors', 'details': validation_response}), 400

        # Pastikan idBall sudah disediakan di new_ball
        idBall = new_ball.get('idBall')
        if idBall is None:
            return jsonify({'success': False, 'message': 'ID ball harus disediakan.'}), 400

        # Load data dengan penanganan kesalahan
        data = load_data()
        if idBall in data:
            return jsonify({'success': False, 'message': 'ID ball sudah ada. Gunakan ID yang unik.'}), 400

        # Save new ball ke data
        data[idBall] = new_ball

        # Simpan data dengan penanganan kesalahan
        save_data(data)
        
        return jsonify({'success': True, 'message': 'Ball created', 'data': data[idBall]}), 201

    except Exception as e:
        # Log kesalahan untuk debugging
        print(f"Error occurred: {str(e)}")
        return jsonify({'success': False, 'message': 'An unexpected error occurred.', 'error': str(e)}), 500

# API untuk mendapatkan semua data bola
@app.route('/api/ball', methods=['GET'])
def get_all_balls():
    try:
        data = load_data()

        # Ubah data bola menjadi array dari objek
        balls = [{"id": ball_id, **ball_data} for ball_id, ball_data in data.items()]

        # Jika tidak ada data bola, kembalikan pesan yang sesuai
        if not balls:
            return jsonify({'success': True, 'message': 'No balls found.', 'data': []}), 200

        # Kembalikan semua data bola dalam bentuk array
        return jsonify({'success': True, 'message': 'Balls retrieved successfully.', 'data': balls}), 200

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'success': False, 'message': 'An error occurred while retrieving balls.', 'error': str(e)}), 500

# API untuk mendapatkan data bola berdasarkan ID
@app.route('/api/ball/<idBall>', methods=['GET'])
def get_ball(idBall):
    data = load_data()
    if idBall in data:
        return jsonify(data[idBall]), 200
    return jsonify({'message': 'Ball not found'}), 404

# API untuk mendapatkan data bola berdasarkan typeArena
@app.route('/api/ball/arena/<type_arena>', methods=['GET'])
def get_balls_by_arena(type_arena):
    if type_arena not in ['a', 'b']:
        return jsonify({'message': 'Invalid arena type: must be "a" or "b"'}), 400

    data = load_data()  # Memuat data mock

    # Filter bola berdasarkan `typeArena`
    filtered_balls = [
        {'idBall': idBall, 'lat': ball['lat'], 'long': ball['long'], 'typeBall': ball['typeBall']}
        for idBall, ball in data.items() if ball['typeArena'] == type_arena
    ]

    if not filtered_balls:
        return jsonify({'message': f'No balls found for arena type: {type_arena}'}), 404

    # Mengembalikan array bola dalam response
    return jsonify(filtered_balls), 200

def detect_from_camera(model):
    up_folder = './up'
    down_folder = './down'
    os.makedirs(up_folder, exist_ok=True)
    os.makedirs(down_folder, exist_ok=True)

    # Variabel untuk melacak confidence tertinggi dan area terbesar
    highest_confidence_up = 0
    largest_area_up = 0

    # Dataset video
    arena_a = 'D:\Kuliah\komunitas\Robotic\Nautronica\ASV-2023\VID_20241023_131021.mp4'
    
    # Capture video
    cap1 = cv2.VideoCapture(arena_a)
    cap2 = cv2.VideoCapture(1)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Failed to open one or both cameras.")
        return

    prev_time = cv2.getTickCount()
    last_emit_time = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Failed to capture frame from one or both cameras.")
            break

        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_time - prev_time)
        prev_time = current_time

        results = model(frame1)
        human_detections = [det for det in results[0].boxes if det.cls[0] == 0]

        for det in human_detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Camera up frame', frame1)
        cv2.imshow('Camera down frame', frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Deteksi manusia berdasarkan confidence tertinggi dan ukuran bounding box terbesar
        if human_detections:
            for det in human_detections:
                # Ambil nilai confidence dan ukuran bounding box
                confidence = det.conf[0]
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                area = (x2 - x1) * (y2 - y1)

                # Cek apakah confidence dan ukuran lebih besar dari nilai sebelumnya
                if confidence > highest_confidence_up and area > largest_area_up:
                    # Perbarui nilai confidence dan area terbesar
                    highest_confidence_up = confidence
                    largest_area_up = area
                    
                    # Simpan gambar dari frame 1 dan frame 2
                    save_best_image(frame1, confidence, up_folder, 'up_image')
                    save_best_image(frame2, confidence, down_folder, 'down_image')

        burst_image_up = encode_image_to_base64(frame1)
        burst_image_down = encode_image_to_base64(frame2)

        if time.time() - last_emit_time >= 1:
            result = {
                'ResultDetection': {
                    'device_up': {'Image': burst_image_up},
                    'device_down': {'Image': burst_image_down}
                }
            }
            sio.emit('ResultDetection', result)
            last_emit_time = time.time()

        time.sleep(0.06)

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

# Fungsi untuk mengirim gambar terbaru ke client setiap 1 detik
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

# Event ketika client connect
@sio.event
def connect(sid, environ):
    print(f'Client connected: {sid}')

if __name__ == '__main__':
    model = YOLO('best.pt')
    app.debug = True,
    sio.start_background_task(detect_from_camera, model=model)
    # sio.start_background_task(detect_from_camera, device='down', model=model)
    # sio.start_background_task(emit_saved_images)
    eventlet.wsgi.server(eventlet.listen(('', 5000)), flask_app)
