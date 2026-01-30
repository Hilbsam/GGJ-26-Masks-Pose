import cv2
from ultralytics import YOLO
import socket

# Netzwerk Setup
UDP_IP = "10.214.244.157"
UDP_PORT = 6666
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 2. Load YOLOv8 Pose model
model = YOLO('yolo26s-pose.pt') 
cap = cv2.VideoCapture(0) # 0 für intern, 1 für extern (ggf. auf 1 ändern)

# Die Indizes für Nase, Oberkörper und Unterkörper (13 Punkte)
SELECTED_POINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # .track() sorgt dafür, dass Personen eine feste ID behalten
    # persist=True speichert die IDs über Frames hinweg
    results = model.track(frame, verbose=False, max_det=2, persist=True)
    
    all_persons_data = []
    
    if results[0].keypoints is not None:
        # Wir iterieren direkt über die gefundenen Personen-Instanzen
        # YOLO trackt hier automatisch, wer Person 1 und wer Person 2 ist
        for p_idx in range(len(results[0].keypoints)):
            kpts = results[0].keypoints.xyn[p_idx].numpy()
            
            # Falls die KI Punkte für diese Person findet
            if len(kpts) > 0:
                filtered_coords = []
                for i in SELECTED_POINTS:
                    # Sicherstellen, dass der Index existiert
                    if i < len(kpts):
                        x, y = kpts[i]
                        filtered_coords.append(f"{x:.4f},{y:.4f}")
                
                all_persons_data.append(",".join(filtered_coords))
    
    # Sende Daten nur, wenn mindestens eine Person gefunden wurde
    if all_persons_data:
        # Format: "P1_x,P1_y...#P2_x,P2_y..."
        final_string = "#".join(all_persons_data)
        sock.sendto(final_string.encode(), (UDP_IP, UDP_PORT))

    # Vorschau-Fenster
    cv2.imshow("YOLO Dual-Pose Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()