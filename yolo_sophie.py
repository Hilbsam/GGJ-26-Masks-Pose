import cv2
from ultralytics import YOLO
import socket

# --- CONFIGURATION ---
DEBUG = True  # Set to False to disable visualization and increase performance
UDP_IP = "10.214.244.157"
UDP_PORT = 6666
MODEL_PATH = 'yolo26s-pose.pt'
SELECTED_POINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Netzwerk Setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Load Model
model = YOLO(MODEL_PATH) 
cap = cv2.VideoCapture(0)

print(f"Starting Pose Tracking... Debug Mode: {'ON' if DEBUG else 'OFF'}")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # Perform tracking (max_det=2 limits detection to two people)
    results = model.track(frame, verbose=False, max_det=2, persist=True)
    
    all_persons_data = []
    
    # Extract Keypoints
    if results[0].keypoints is not None:
        for p_idx in range(len(results[0].keypoints)):
            kpts = results[0].keypoints.xyn[p_idx].numpy()
            
            if len(kpts) > 0:
                filtered_coords = []
                for i in SELECTED_POINTS:
                    if i < len(kpts):
                        x, y = kpts[i]
                        filtered_coords.append(f"{x:.4f},{y:.4f}")
                
                all_persons_data.append(",".join(filtered_coords))
    
    # Send UDP Data
    if all_persons_data:
        final_string = "#".join(all_persons_data)
        sock.sendto(final_string.encode(), (UDP_IP, UDP_PORT))

    # --- DEBUG / VISUALIZATION ---
    if DEBUG:
        # results[0].plot() draws the skeleton, IDs, and boxes
        annotated_frame = results[0].plot() 
        cv2.imshow("YOLO Debug View", annotated_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # In non-debug mode, we still need a tiny waitKey to process events
        # but we don't render anything to the screen
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()