# import socketio
# import json
# import websockets
# import eventlet
# import piexif
# import time
# import base64
# import cv2
# from PIL import Image
# from io import BytesIO
# from ultralytics import YOLO
# import eventlet.wsgi

# # Inisialisasi Socket.IO
# sio = socketio.Server(cors_allowed_origins='*')
# app = socketio.WSGIApp(sio)

# # Variabel global untuk menyimpan data MAVLink
# mavlink_data = {
#     'latitude': None,
#     'longitude': None,
#     'sog': None,
#     'cog': None
# }

# def get_mavlink_data():
#     """Mengambil data MAVLink dari WebSocket."""
#     url = f"ws://localhost:6040/v1/ws/mavlink"
#     try:
#         with eventlet.greenio.green.websocket.websocket_connect(url) as websocket:
#             print("Connected to MAVLink WebSocket")
#             while True:
#                 message = websocket.recv()  # Terima pesan dari WebSocket
#                 data = json.loads(message)  # Parse pesan JSON

#                 if data.get("message", {}).get("type") == "GLOBAL_POSITION_INT":
#                     # Simpan data MAVLink ke variabel global
#                     mavlink_data['latitude'] = data["message"]["lat"] / 1e7
#                     mavlink_data['longitude'] = data["message"]["lon"] / 1e7
#                     mavlink_data['cog'] = data["message"]["hdg"] / 100  # Dalam derajat
#                     mavlink_data['sog'] = data["message"]["vx"] / 100  # Dalam m/s
#                     print(f"Lat: {mavlink_data['latitude']}, Lng: {mavlink_data['longitude']}, "
#                           f"COG: {mavlink_data['cog']}, SOG: {mavlink_data['sog']}")

#     except Exception as e:
#         print(f"WebSocket connection error: {e}")

# def convert_to_dms(value):
#     if value is None:
#         raise ValueError("Latitude or Longitude value is None.")
    
#     degrees = int(value)
#     minutes = int((value - degrees) * 60)
#     seconds = (value - degrees - minutes / 60) * 3600
#     return ((degrees, 1), (minutes, 1), (int(seconds * 100), 100))

# def generate_exif(lat=None, lon=None, sog=None, cog=None, day=None, date=None, time_str=None):
#     """Membuat metadata EXIF untuk gambar."""
    
#     # Set default value jika tidak ada GPS data
#     lat = lat if lat is not None else 0.0
#     lon = lon if lon is not None else 0.0

#     exif_dict = {
#         "0th": {
#             piexif.ImageIFD.Make: "Custom Camera",
#             piexif.ImageIFD.XPTitle: f"SOG: {sog} knot, COG: {cog}°".encode('utf-16'),
#         },
#         "Exif": {
#             piexif.ExifIFD.DateTimeOriginal: f"{date} {time_str}",
#             piexif.ExifIFD.UserComment: f"Day: {day}".encode('utf-16'),
#         },
#         "GPS": {
#             piexif.GPSIFD.GPSLatitude: convert_to_dms(lat),
#             piexif.GPSIFD.GPSLongitude: convert_to_dms(lon),
#         }
#     }

#     return piexif.dump(exif_dict)

# def encode_image_to_base64(image, mavlink_data):
#     """Mengubah gambar ke Base64 dan menambahkan metadata MAVLink."""
#     pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # Ambil data MAVLink dan waktu saat ini
#     lat = mavlink_data.get('latitude', 0.0)
#     lon = mavlink_data.get('longitude', 0.0)
#     sog = mavlink_data.get('sog', 0.0)
#     cog = mavlink_data.get('cog', 0.0)
    
#     # Mengambil waktu hanya sekali
#     current_time = time.strftime("%a %d/%m/%Y %H:%M:%S")
#     day, date, time_str = current_time.split()

#     # Buat metadata EXIF
#     exif_bytes = generate_exif(lat, lon, sog, cog, day, date, time_str)

#     # Simpan gambar ke buffer dengan metadata
#     buffer = BytesIO()
#     pil_img.save(buffer, format="JPEG", exif=exif_bytes)
#     buffer.seek(0)

#     # Encode gambar ke Base64
#     return base64.b64encode(buffer.read()).decode('utf-8')

# def detect_objects_from_frame(model, frame):
#     """Deteksi objek menggunakan YOLO."""
#     results = model(frame)
#     detections = []

#     for result in results[0].boxes:
#         label = result.cls  # Index kelas
#         confidence = result.conf.item()  # Akurasi (confidence)
#         label_name = model.names[int(label)]  # Konversi index ke nama kelas

#         detections.append({
#             'Label': label_name,
#             'Accuracy': round(confidence, 2)
#         })

#     return detections

# def detect_from_camera(device):
#     """Mendeteksi objek dan mengirim data beserta gambar."""
#     model = YOLO('yolov8m.pt')
#     cap = cv2.VideoCapture(0 if device == 'cam1' else 1)  # Pilih webcam

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Deteksi objek dan encode gambar dengan metadata MAVLink terbaru
#         detections_cam = detect_objects_from_frame(model, frame)
#         image_base64 = encode_image_to_base64(frame, mavlink_data)

#         result = {
#             'ResultDetection': {
#                 device: {
#                     'Image': image_base64,
#                     'Detections': detections_cam,
#                     'Position': mavlink_data  # Tambahkan MAVLink data
#                 }
#             }
#         }

#         # Kirim data ke client melalui Socket.IO
#         sio.emit('ResultDetection', result)
#         time.sleep(0.1)  # Atur jeda agar tidak terlalu cepat

#     cap.release()

# @sio.event
# def connect(sid, environ):
#     """Handle client connect."""
#     print(f'Client connected: {sid}')

# @sio.event
# def disconnect(sid):
#     """Handle client disconnect."""
#     print(f'Client disconnected: {sid}')

# if __name__ == '__main__':
#     # Konfigurasi untuk koneksi MAVLink
#     ip_public = "localhost"  # Ganti dengan IP publik MAVLink
#     port_mavlink = 6040  # Ganti dengan port MAVLink
#     camera_devices = ['cam1', 'cam2']  # Daftar kamera yang digunakan

#     # Jalankan pengambilan data MAVLink dan kamera secara paralel
#     eventlet.spawn_n(get_mavlink_data)
#     for device in camera_devices:
#         sio.start_background_task(detect_from_camera, device)

#     # Jalankan server WSGI untuk Socket.IO
#     server = eventlet.listen(('', 5000))
#     eventlet.wsgi.server(server, app)

import eventlet
eventlet.monkey_patch()
import socketio
import cv2
import base64
import time
import torch
from ultralytics import YOLO

# Inisialisasi Socket.IO
sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio)

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Fungsi deteksi multi-objek
def detect_objects_from_frame(model, frame):
    results = model(frame)
    detections = []

    for result in results[0].boxes:
        label = result.cls  # Index kelas
        confidence = result.conf.item()  # Akurasi (confidence)
        label_name = model.names[int(label)]  # Konversi index ke nama kelas

        detections.append({
            'Label': label_name,
            'Accuracy': round(confidence, 2)
        })

    return detections

def detect_from_camera(device):
    model = YOLO('yolov8m.pt')
    cap = cv2.VideoCapture(0 if device == 'cam1' else 1)  # Gunakan 2 webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections_cam = detect_objects_from_frame(model, frame)
        image_base64 = encode_image_to_base64(frame)

        result = {
            'ResultDetection': {
                device: {
                    'Image': image_base64,
                    'Detections': detections_cam  # List deteksi multi-objek
                }
            }
        }

        sio.emit('ResultDetection', result)
        time.sleep(0.1)  # Atur jeda agar tidak terlalu cepat

    cap.release()

@sio.event
def connect(sid, environ):
    print(f'Client connected: {sid}')

@sio.event
def disconnect(sid):
    print(f'Client disconnected: {sid}')

if __name__ == '__main__':
    sio.start_background_task(detect_from_camera, 'cam1')
    sio.start_background_task(detect_from_camera, 'cam2')
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
