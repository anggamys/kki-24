import socketio
import eventlet
import cv2
import base64
import time
import threading
import signal
import sys
import os
from ultralytics import YOLO
import numpy as np

# Load model YOLOv8
model = YOLO('yolov8m.pt')

# Initialize Socket.IO server
sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio)

# Variables to track state
last_emit_time = time.time()
camera_running = True  # Flag to stop camera detection

# Directory to save detected images
save_dir = 'detected_images'
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Path to the current best detected image
best_image_path = os.path.join(save_dir, 'best_detected.jpg')
best_score = 0  # Initialize the best detection score

# Function to encode image to base64 from a file
def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Event when client connects
@sio.event
def connect(sid, environ):
    print(f'Client connected: {sid}')

# Event when client disconnects
@sio.event
def disconnect(sid):
    print(f'Client disconnected: {sid}')

# Function for object detection from camera frames
def detect_from_camera():
    global last_emit_time, camera_running, best_score

    # Open the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open camera")
        return

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        current_time = time.time()
        
        # Perform detection every 2 seconds
        output = model(frame)
        detected = False
        current_best_score = 0  # Track the current best score for this frame

        for result in output:
            if result.boxes is not None:
                for box in result.boxes.data.tolist():
                    class_id = int(box[5])  # Assuming class_id is at index 5
                    score = box[4]  # Assuming confidence score is at index 4
                    if class_id == 0:  # Check for 'person'
                        detected = True
                        current_best_score = max(current_best_score, score)  # Update current best score

        if detected and current_best_score > best_score:
            best_score = current_best_score  # Update the best score
            result_image = output[0].plot()  # Image with bounding box
            
            # Save the detected image with the highest score
            cv2.imwrite(best_image_path, result_image)  # Save as JPG

        # Send the image every second
        if current_time - last_emit_time >= 1:
            if os.path.exists(best_image_path):
                # Encode the saved image to base64
                image_64 = encode_image_to_base64(best_image_path)

                # Emit image to client with acknowledgment
                sio.emit('image', {'image': image_64}, callback=lambda data: print('Client acknowledged:', data))

                print(f"Image sent with size: {len(image_64)} bytes")

            last_emit_time = current_time  # Update time after sending

        # Allow OpenCV to update the window
        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit if 'q' is pressed

    cap.release()
    cv2.destroyAllWindows()

# Function to run the server
def run_server():
    print("Starting Socket.IO server...")
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
    print("Socket.IO server running at http://localhost:5000")

# Signal handler to gracefully stop the program
def signal_handler(sig, frame):
    global camera_running
    print("Stopping program...")
    camera_running = False  # Set flag to stop camera detection
    sys.exit(0)  # Exit the program

# Run the server and camera detection
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    # Start camera detection in a separate thread
    camera_thread = threading.Thread(target=detect_from_camera)
    camera_thread.start()
    
    # Run Socket.IO server in the main thread
    run_server()
