import cv2
from ultralytics import YOLO
import socket

# 1. Setup Network (IP and Port)
UDP_IP = "127.0.0.1"
UDP_PORT = 6666
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 2. Load YOLOv8 Pose model
model = YOLO('yolov8n-pose.pt') 
cap = cv2.VideoCapture(0) # Change to 1 or 2 for external camera

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, verbose=False)

    for r in results:
        if r.keypoints:
            # Get normalized (0 to 1) coordinates
            kpts = r.keypoints.xyn[0].numpy() 
            # Create a string "x,y,x,y,..."
            data = ",".join([f"{kp[0]:.4f},{kp[1]:.4f}" for kp in kpts])
            sock.sendto(data.encode(), (UDP_IP, UDP_PORT))

    cv2.imshow("Python Sender", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()