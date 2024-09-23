from ultralytics import YOLO
import time
import cv2
import math 

model = YOLO("./yolov8m.pt")

prev_time = time.time()
fps = 0
# start webcam
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
    results = model(frame)

    annotated_frame = results[0].plot()


    # Tampilkan FPS di pojok kiri atas
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    print(fps,"fps \n")

    # Tampilkan frame yang sedang diproses (opsional)
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Jika 'q' ditekan, keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
