import socket
import json
from ultralytics import YOLO
import cv2

# UDP Configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 6666
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Load Model
model = YOLO("yYOLO26s-pose.pt")

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run Inference (stream=True for memory efficiency)
    results = model(frame, stream=True, verbose=False)

    for r in results:
        if r.keypoints is not None:
            # Convert keypoints to a list of [x, y] coordinates
            # We normalize or send raw pixels (here we send raw pixels)
            points = r.keypoints.xy.cpu().numpy().tolist()
            
            if len(points) > 0:
                # Send the first detected person
                data = json.dumps(points[0])
                sock.sendto(data.encode(), (UDP_IP, UDP_PORT))

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()