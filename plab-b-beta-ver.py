from ultralytics import YOLO
import cv2

# Parameters
output_width = 640
output_height = 480
CONFIDENCE_THRESHOLD = 0.7
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# Load the YOLO model
model = YOLO(r"E:\Code\ASV\bola.pt").to('cpu')

# List of image paths
image_paths = [
    r"C:\Users\Vannn\Documents\GomPlayer\Capture\VID_20241023_131409.mp4_000198799.png",
    r"C:\Users\Vannn\Documents\GomPlayer\Capture\VID_20241023_131409.mp4_000194411.png",
    r"C:\Users\Vannn\Documents\GomPlayer\Capture\VID_20241023_131409.mp4_000195442.png",
    r"C:\Users\Vannn\Documents\GomPlayer\Capture\VID_20241023_131409.mp4_000196715.png",
    r"C:\Users\Vannn\Documents\GomPlayer\Capture\VID_20241023_131409.mp4_000197185.png",
    r"C:\Users\Vannn\Documents\GomPlayer\Capture\VID_20241023_131409.mp4_000197643.png",
    r"C:\Users\Vannn\Documents\GomPlayer\Capture\VID_20241023_131409.mp4_000240531.png",
    r"C:\Users\Vannn\Documents\GomPlayer\Capture\VID_20241023_131409.mp4_000234700.png",
    r"C:\Users\Vannn\Documents\GomPlayer\Capture\VID_20241023_131409.mp4_000235397.png"
]

# Process each image
for img_path in image_paths:
    # Read the image
    image = cv2.imread(img_path)
    image = cv2.resize(image, (output_width, output_height))  # Resize image

    # Run inference on the image
    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, imgsz=(output_width, output_height))

    max_red_area = 0
    max_green_area = 0
    bgreen_tensor = None
    bred_tensor = None

    # Loop through the detected objects
    for box in results[0].boxes:
        w = box.xywh[0][2]
        h = box.xywh[0][3]
        area = max(w * w, h * h)

        # Determine if the object is "green" or "red"
        if int(box.cls[0]) == 1:  # Green class (adjust according to your model)
            if max_green_area < area:
                max_green_area = area
                bgreen_tensor = box
        elif int(box.cls[0]) == 0:  # Red class (adjust according to your model)
            if max_red_area < area:
                max_red_area = area
                bred_tensor = box

    # If a green object is detected, draw a rectangle around it
    if bgreen_tensor is not None:
        x1_green, y1_green, x2_green, y2_green = map(int, bgreen_tensor.xyxy[0])
        cv2.rectangle(image, (x1_green, y1_green), (x2_green, y2_green), GREEN, 2)

    # If a red object is detected, draw a rectangle around it
    if bred_tensor is not None:
        x1_red, y1_red, x2_red, y2_red = map(int, bred_tensor.xyxy[0])
        cv2.rectangle(image, (x1_red, y1_red), (x2_red, y2_red), RED, 2)

    # If both green and red objects are detected, calculate and draw the midpoint and lines
    if bgreen_tensor and bred_tensor:
        # Bottom center of the frame
        bottom_x_y = (output_width // 2, output_height)

        # Calculate the midpoint between the top centers of the red and green boxes
        line_x_top_center = (x2_green + x1_red) // 2
        line_y_top_center = (y1_green + y1_red) // 2
        line_center_x_y = (line_x_top_center, line_y_top_center)

        # Draw a line between the green box's top-right and the red box's top-left corners
        cv2.line(image, (x2_green, y1_green), (x1_red, y1_red), color=(225, 0, 0), thickness=2)

        # Draw a line from the bottom center to the calculated midpoint
        cv2.line(image, bottom_x_y, line_center_x_y, thickness=2, color=(255, 254, 254))

    # Display the image with the detections
    cv2.imshow('Detections', image)

    # Save the processed image with detections (optional)
    output_path = img_path.replace(".png", "_detected.png")
    cv2.imwrite(output_path, image)

    # Wait for a key press and close the image window
    cv2.waitKey(0)

# Close any open windows
cv2.destroyAllWindows()
