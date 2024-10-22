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
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading

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
def save_best_image(image, confidence, folder_path, filename):
    file_path = os.path.join(folder_path, filename)
    cv2.imwrite(file_path, image)
    print(f"Image saved at {file_path} with confidence {confidence:.2f}")

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

# Fungsi untuk mendeteksi manusia dari frame kamera
def detect_from_camera(device, model):
    up_folder = './up'
    down_folder = './down'
    os.makedirs(up_folder, exist_ok=True)
    os.makedirs(down_folder, exist_ok=True)

    highest_confidence_up = 0
    highest_confidence_down = 0

    source = 0 if device == 'up' else 1
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open camera: {device}")
        return

    prev_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture frame from {device}")
            break

        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_time - prev_time)
        prev_time = current_time

        results = model(frame)
        human_detections = [det for det in results[0].boxes if det.cls[0] == 0]

        for det in human_detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow(f'{device} frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if human_detections:
            best_detection = max(human_detections, key=lambda det: det.conf[0])
            confidence = best_detection.conf[0]

            if device == "up" and confidence > highest_confidence_up:
                highest_confidence_up = confidence
                save_best_image(frame, confidence, up_folder, 'best_up.jpg')

            elif device == 'down' and confidence > highest_confidence_down:
                highest_confidence_down = confidence
                save_best_image(frame, confidence, down_folder, 'best_down.jpg')

        burst_image = encode_image_to_base64(frame)

        result = {
            'ResultDetection': {
                device: {
                    'Image': burst_image
                }
            }
        }
        sio.emit('ResultDetection', result)
        time.sleep(0.06)

    cap.release()
    cv2.destroyAllWindows()

# Fungsi untuk mengirim gambar yang tersimpan ke client setiap 1 detik
def emit_saved_images():
    while True:
        up_image_path = './up/best_up.jpg'
        if os.path.exists(up_image_path):
            image_64 = encode_image_to_base64(cv2.imread(up_image_path))
            sio.emit('up', {'data': image_64})

        down_image_path = './down/best_down.jpg'
        if os.path.exists(down_image_path):
            image_64 = encode_image_to_base64(cv2.imread(down_image_path))
            sio.emit('down', {'data': image_64})

        time.sleep(1)

# Event ketika client connect
@sio.event
def connect(sid, environ):
    print(f'Client connected: {sid}')

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')
    app.debug = True,
    # sio.start_background_task(detect_from_camera, device='up', model=model)
    # sio.start_background_task(detect_from_camera, device='down', model=model)
    # sio.start_background_task(emit_saved_images)
    eventlet.wsgi.server(eventlet.listen(('', 5000)), flask_app)
