import cv2
from ultralytics import YOLO
import socket
import numpy as np

# --- CONFIGURATION ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
MODEL_PATH = 'yolo26s-pose.pt' # Consider using the 'n' (nano) version if speed > accuracy
IMG_SIZE = 320  # Reduce resolution for massive speed boost (e.g., 320, 480, or 640)
SELECTED_POINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Network Setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Load Model - Setting task explicitly helps
model = YOLO(MODEL_PATH)

# Camera Setup
cap = cv2.VideoCapture(0)
# Lowering camera resolution here reduces CPU/USB bandwidth
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce lag by not buffering old frames

print(f"Starting Headless Pose Tracking at {IMG_SIZE}px...")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # 1. Use .predict instead of .track if you don't need persistent IDs 
        # (Since your Unity script sorts by X-position anyway, you don't need tracking!)
        # imgsz=IMG_SIZE is the biggest speed factor. 
        results = model.predict(
            frame, 
            conf=0.5, 
            imgsz=IMG_SIZE, 
            verbose=False, 
            max_det=4, 
            stream=True # stream=True is more memory efficient
        )
        
        for r in results:
            if r.keypoints is not None:
                # Use .xyn (normalized) and move to cpu once
                kpts_all = r.keypoints.xyn.cpu().numpy()
                
                all_persons_data = []
                for person_kpts in kpts_all:
                    if len(person_kpts) > 0:
                        # Vectorized slicing is faster than manual loops
                        filtered = person_kpts[SELECTED_POINTS]
                        
                        # Faster string joining: map float to string with limited precision
                        # This replaces the nested for-loops and multiple joins
                        coords_str = ",".join(map(lambda p: f"{p[0]:.3f},{p[1]:.3f}", filtered))
                        all_persons_data.append(coords_str)

                if all_persons_data:
                    final_string = "#".join(all_persons_data)
                    sock.sendto(final_string.encode(), (UDP_IP, UDP_PORT))

except KeyboardInterrupt:
    print("Stopping...")

cap.release()
cv2.destroyAllWindows()