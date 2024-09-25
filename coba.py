import cv2 as cv
vcap = cv.VideoCapture("rtsp://172.25.165.51:554/live")
while(1):
    ret, frame = vcap.read()
    cv.imshow('VIDEO', frame)
    cv.waitKey(1)