# import cv2
# import time
# import requests
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# import pygame
# from pygame import mixer
# import torch
# import mediapipe as mp

# import sys
# import os

# # Add root directory to Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # Initialize
# mixer.init()
# pygame.init()

# # Configuration
# API_ENDPOINT = "http://localhost:8000/process-frame"
# MUSCLE_GROUP = "BACK"  # Hardcoded for example
# EXERCISE = "deadlifts"     # Hardcoded for example
# # YOLO_MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# YOLO_MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False, _verbose=False)
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# def draw_countdown(frame, count):
#     img = Image.fromarray(frame)
#     draw = ImageDraw.Draw(img)
#     font = ImageFont.truetype("arial.ttf", 200)
    
#     # Draw countdown
#     text = str(count)
#     w, h = draw.textsize(text, font=font)
#     draw.text(
#         ((img.width-w)/2, (img.height-h)/2), 
#         text, 
#         font=font, 
#         fill=(0,255,0,0)
#     )
    
#     # Play sound
#     mixer.Sound('beep.wav').play()
#     return np.array(img)

# def process_frame(frame):
#     # YOLOv5 person detection
#     results = YOLO_MODEL(frame)
#     persons = results.xyxy[0][results.xyxy[0][:, 5] == 0]  # class 0 = person
    
#     if len(persons) > 0:
#         # Get largest person
#         max_area = 0
#         best_box = None
#         for box in persons:
#             x1, y1, x2, y2, conf, _ = box
#             area = (x2-x1)*(y2-y1)
#             if area > max_area:
#                 max_area = area
#                 best_box = box
        
#         # Expand bounding box by 20%
#         x1, y1, x2, y2 = best_box[:4]
#         w = x2 - x1
#         h = y2 - y1
#         x1 = max(0, x1 - w*0.1)
#         y1 = max(0, y1 - h*0.1)
#         x2 = min(frame.shape[1], x2 + w*0.1)
#         y2 = min(frame.shape[0], y2 + h*0.1)
        
#         # Crop frame
#         cropped = frame[int(y1):int(y2), int(x1):int(x2)]
        
#         # MediaPipe processing
#         results = pose.process(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
#         if results.pose_landmarks:
#             keypoints = []
#             for landmark in results.pose_landmarks.landmark:
#                 keypoints.append([
#                     landmark.x, 
#                     landmark.y, 
#                     landmark.z
#                 ])
            
#             return keypoints
        
#     return None

# def main():
#     cap = cv2.VideoCapture(0)
    
#     # Countdown timer
#     for i in range(5, 0, -1):
#         ret, frame = cap.read()
#         if ret:
#             frame = draw_countdown(frame, i)
#             cv2.imshow('Posture Analysis', frame)
#             cv2.waitKey(1000)
    
#     # Main loop
#     while True:
#         start_time = time.time()
#         ret, frame = cap.read()
        
#         if ret:
#             # Process frame
#             keypoints = process_frame(frame)
            
#             if keypoints:
#                 # Send to backend
#                 response = requests.post(
#                     API_ENDPOINT,
#                     json={
#                         "keypoints": keypoints,
#                         "muscle_group": MUSCLE_GROUP,
#                         "exercise": EXERCISE,
#                         "timestamp": time.time()
#                     }
#                 )
                
#                 if response.status_code == 200:
#                     result = response.json()
#                     # Display results
#                     if 'class' in result:
#                         label = "Correct" if result['class'] == 0 else "Incorrect"
#                         cv2.putText(
#                             frame,
#                             f"{label} ({result['confidence']:.2f})",
#                             (10, 30), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 
#                             1, 
#                             (0,255,0), 
#                             2
#                         )
                        
#                         # Audio feedback
#                         if result['class'] != 0:
#                             # mixer.Sound('alert.wav').play()
#                             mixer.Sound('assets/sounds/beep.wav').play()
#                             print("Prediction Results: ", result['prediction'])
#                             print("Prediction Class", result['class'])
            
#             # Maintain target FPS
#             processing_time = time.time() - start_time
#             wait_time = max(1, int(1000/30 - processing_time*1000))
            
#             cv2.imshow('Posture Analysis', frame)
#             if cv2.waitKey(wait_time) & 0xFF == ord('q'):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()




import cv2
import time
import requests
import mediapipe as mp
import torch
import numpy as np
import winsound
from collections import deque

# Configuration
MODEL_URL = "http://192.168.1.5:5000/predict"
BUFFER_SIZE = 150  # 5 seconds at 30fps
TARGET_FPS = 30
MARGIN_RATIO = 0.2  # 20% padding around detected person

# Load YOLOv5
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def draw_countdown(frame, count):
    h, w = frame.shape[:2]
    cv2.putText(frame, str(count), (w//2-50, h//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 8)
    winsound.Beep(1000, 500)

def process_frame(frame):
    # YOLOv5 person detection
    results = yolo(frame)
    persons = results.xyxy[0][results.xyxy[0][:, 5] == 0]  # class 0 = person
    
    if len(persons) > 0:
        # Get largest person
        person = max(persons, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
        x1, y1, x2, y2 = map(int, person[:4])
        
        # Add padding
        w = x2 - x1
        h = y2 - y1
        x1 = max(0, x1 - int(w * MARGIN_RATIO))
        y1 = max(0, y1 - int(h * MARGIN_RATIO))
        x2 = min(frame.shape[1], x2 + int(w * MARGIN_RATIO))
        y2 = min(frame.shape[0], y2 + int(h * MARGIN_RATIO))
        
        # Crop and process with MediaPipe
        cropped = frame[y1:y2, x1:x2]
        results = pose.process(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
            return landmarks, cropped
    
    return None, frame

def main():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera FPS: {fps}")
    
    # Countdown sequence
    for i in range(5, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        draw_countdown(frame, i)
        cv2.imshow('Posture Check', frame)
        cv2.waitKey(1)
        time.sleep(1)
    
    # Main loop
    last_time = time.time()
    frame_times = deque(maxlen=30)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        landmarks, processed_frame = process_frame(frame)
        
        # Calculate real FPS
        frame_times.append(time.time())
        if len(frame_times) > 1:
            real_fps = len(frame_times) / (frame_times[-1] - frame_times[0])
        else:
            real_fps = fps
        
        if landmarks:
            # Send to backend
            data = {
                'keypoints': landmarks,
                'muscle_group': 'BACK',  # Hardcoded for demo
                'exercise': 'deadlifts',  # Hardcoded for demo
                'fps': real_fps
            }
            
            try:
                response = requests.post(MODEL_URL, json=data).json()
                if 'result' in response:
                    cv2.putText(processed_frame,
                               f"{response['result'].upper()} ({response.get('error', '')})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                               (0, 255, 0) if response['result'] == 'correct' else (0, 0, 255), 2)
            except Exception as e:
                print(f"API Error: {e}")
        
        # Show processed frame
        cv2.imshow('Posture Check', processed_frame)
        
        # Maintain target FPS
        elapsed = time.time() - last_time
        wait_time = max(1, int((1/TARGET_FPS - elapsed)*1000))
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
        
        last_time = time.time()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()