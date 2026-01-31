import cv2
import socket
import time
import torch
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

# --- CONFIGURATION ---
DEBUG = True
UDP_IP = "10.214.244.157"
UDP_PORT = 6666
MODEL_PATH = 'yolo26s-pose.pt'
SELECTED_POINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
FACE_TIMEOUT = 60.0  # Keep ID for this many seconds after disappears
FACE_MATCH_THRESHOLD = 0.8  # Euclidean distance threshold (lower is stricter)

# Network Setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Load Models
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

print("Loading FaceNet model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print(f"Starting Pose Tracking with Face ID... Device: {device}")


class FaceTracker:
    def __init__(self, timeout=60.0, threshold=0.7):
        # Dictionary of {internal_id: {'embedding': tensor, 'last_seen': timestamp}}
        self.known_faces = []
        self.next_id = 1
        self.timeout = timeout
        self.threshold = threshold
        
        # Cache to map current YOLO track IDs to verified internal IDs
        # {yolo_id: internal_id}
        self.yolo_id_map = {}

    def get_identity(self, embedding, yolo_id):
        current_time = time.time()
        
        # 1. Clean up old faces from memory
        self.known_faces = [f for f in self.known_faces if current_time - f['last_seen'] < self.timeout]
        
        # 2. If we have a mapped ID for this YOLO track, return it (Optimization)
        # However, if YOLO ID switches (the user's problem), this function is called with a NEW yolo_id.
        # So we only use the map if we trust the yolo_id is stable for a bit.
        # But we should also handle if embedding is None (face hidden) - then rely on yolo_id.
        
        if embedding is None:
            # Fallback: if we know this yolo_id, use the associated internal_id
            if yolo_id is not None and yolo_id in self.yolo_id_map:
                internal_id = self.yolo_id_map[yolo_id]
                # Update timestamp for this face
                for f in self.known_faces:
                    if f['id'] == internal_id:
                        f['last_seen'] = current_time
                        break
                return internal_id
            else:
                # No face, unknown YOLO ID -> Assign temporary new ID or negative? 
                # We'll just return a new ID effectively, but not save it in face DB without embedding
                return -1 

        # 3. We have an embedding. Try to match with known faces
        best_match_id = -1
        min_dist = float('inf')
        
        for face in self.known_faces:
            dist = (face['embedding'] - embedding).norm().item()
            if dist < min_dist:
                min_dist = dist
                best_match_id = face['id']

        if min_dist < self.threshold:
            # Match found!
            matched_id = best_match_id
            # Update last known time
            for face in self.known_faces:
                if face['id'] == matched_id:
                    face['last_seen'] = current_time
                    break
        else:
            # No match found, create new ID
            matched_id = self.next_id
            self.next_id += 1
            self.known_faces.append({
                'id': matched_id,
                'embedding': embedding,
                'last_seen': current_time
            })
            
        # 4. Update the YOLO map
        if yolo_id is not None:
            self.yolo_id_map[yolo_id] = matched_id
            
        return matched_id

tracker = FaceTracker(timeout=FACE_TIMEOUT, threshold=FACE_MATCH_THRESHOLD)

def get_face_embedding(frame, kpts):
    """
    Extract face embedding from frame using pose keypoints for cropping.
    kpts: tensor or numpy array of specific user [17, 2] (normalized or pixels)
    """
    # Keypoints: 0=Nose, 1=LEye, 2=REye, 3=LEar, 4=REar
    # Check if we have detected face points
    # Note: kpts come as normalized xy in current code logic below
    
    # We need pixel coordinates for cropping
    h, w, _ = frame.shape
    
    # Let's get pixel coords for the first 5 keypoints
    face_kpts = []
    # kpts is expected to be shape (17, 2) or similar
    # In YOLO output, 0,0 is usually (0,0) if not detected.
    
    for i in range(5):
        if i < len(kpts):
            kx, ky = kpts[i]
            # YOLO normalized coords might be 0 if not detected or just valid float
            if kx > 0 and ky > 0: 
                face_kpts.append([kx * w, ky * h])
    
    if len(face_kpts) < 2: # Need at least a few points to define a box
        return None
        
    face_kpts = np.array(face_kpts)
    min_x = np.min(face_kpts[:, 0])
    max_x = np.max(face_kpts[:, 0])
    min_y = np.min(face_kpts[:, 1])
    max_y = np.max(face_kpts[:, 1])
    
    # Add padding
    pad_x = (max_x - min_x) * 0.5
    pad_y = (max_y - min_y) * 0.5
    
    x1 = int(max(0, min_x - pad_x))
    y1 = int(max(0, min_y - pad_y))
    x2 = int(min(w, max_x + pad_x))
    y2 = int(min(h, max_y + pad_y))
    
    if x2 - x1 < 20 or y2 - y1 < 20: # Too small
        return None
        
    face_crop = frame[y1:y2, x1:x2]
    
    try:
        # Preprocess for FaceNet
        face_crop = cv2.resize(face_crop, (160, 160))
        face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float().to(device)
        face_tensor = (face_tensor - 127.5) / 128.0 # Normalize -1 to 1
        face_tensor = face_tensor.unsqueeze(0) # Batch dimension
        
        with torch.no_grad():
            embedding = resnet(face_tensor)
            
        return embedding[0].cpu()
    except Exception as e:
        print(f"Face processing error: {e}")
        return None


cap = cv2.VideoCapture(0)

# Main Loop
while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # YOLO Tracking
    results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False, max_det=4)
    
    if DEBUG:
        annotated_frame = results[0].plot()
    
    all_persons_data = []
    
    if results[0].boxes is not None and results[0].keypoints is not None:
        boxes = results[0].boxes
        keypoints = results[0].keypoints
        
        for i, box in enumerate(boxes):
            # Get YOLO ID
            yolo_id = int(box.id.item()) if box.id is not None else None
            
            # Get Keypoints (normalized for output, pixels for cropping)
            kpts_norm = keypoints.xyn[i].cpu().numpy()
            
            # Try to get Face Embedding
            embedding = get_face_embedding(frame, kpts_norm)
            
            # Resolve Identity
            final_id = tracker.get_identity(embedding, yolo_id)
            
            # Prepare Data String
            if len(kpts_norm) > 0:
                filtered_coords = [str(final_id)] # Prepend ID
                for kidx in SELECTED_POINTS:
                    if kidx < len(kpts_norm):
                        x, y = kpts_norm[kidx]
                        filtered_coords.append(f"{x:.4f},{y:.4f}")
                
                all_persons_data.append(",".join(filtered_coords))
            
            # Visualization override
            if DEBUG and final_id != -1:
                # Draw Custom ID on annotated_frame (overwriting YOLO ID logic potentially)
                box_coords = box.xyxy[0].cpu().numpy()
                cx = int(box_coords[0])
                cy = int(box_coords[1])
                # Draw a background for text for better visibility
                text = f"PID: {final_id}"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated_frame, (cx, cy - 30), (cx + w, cy), (0, 0, 255), -1)
                cv2.putText(annotated_frame, text, (cx, cy - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Send UDP Data
    if all_persons_data:
        final_string = "#".join(all_persons_data)
        sock.sendto(final_string.encode(), (UDP_IP, UDP_PORT))

    # --- DEBUG / VISUALIZATION ---
    if DEBUG:
        cv2.imshow("YOLO Debug View", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()