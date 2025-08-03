# # import cv2
# # import time
# # import requests
# # import numpy as np
# # from PIL import Image, ImageDraw, ImageFont
# # import pygame
# # from pygame import mixer
# # import torch
# # import mediapipe as mp

# # import sys
# # import os

# # # Add root directory to Python path
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # # Initialize
# # mixer.init()
# # pygame.init()

# # # Configuration
# # API_ENDPOINT = "http://localhost:8000/process-frame"
# # MUSCLE_GROUP = "BACK"  # Hardcoded for example
# # EXERCISE = "deadlifts"     # Hardcoded for example
# # # YOLO_MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# # YOLO_MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False, _verbose=False)
# # mp_pose = mp.solutions.pose
# # pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # def draw_countdown(frame, count):
# #     img = Image.fromarray(frame)
# #     draw = ImageDraw.Draw(img)
# #     font = ImageFont.truetype("arial.ttf", 200)
    
# #     # Draw countdown
# #     text = str(count)
# #     w, h = draw.textsize(text, font=font)
# #     draw.text(
# #         ((img.width-w)/2, (img.height-h)/2), 
# #         text, 
# #         font=font, 
# #         fill=(0,255,0,0)
# #     )
    
# #     # Play sound
# #     mixer.Sound('beep.wav').play()
# #     return np.array(img)

# # def process_frame(frame):
# #     # YOLOv5 person detection
# #     results = YOLO_MODEL(frame)
# #     persons = results.xyxy[0][results.xyxy[0][:, 5] == 0]  # class 0 = person
    
# #     if len(persons) > 0:
# #         # Get largest person
# #         max_area = 0
# #         best_box = None
# #         for box in persons:
# #             x1, y1, x2, y2, conf, _ = box
# #             area = (x2-x1)*(y2-y1)
# #             if area > max_area:
# #                 max_area = area
# #                 best_box = box
        
# #         # Expand bounding box by 20%
# #         x1, y1, x2, y2 = best_box[:4]
# #         w = x2 - x1
# #         h = y2 - y1
# #         x1 = max(0, x1 - w*0.1)
# #         y1 = max(0, y1 - h*0.1)
# #         x2 = min(frame.shape[1], x2 + w*0.1)
# #         y2 = min(frame.shape[0], y2 + h*0.1)
        
# #         # Crop frame
# #         cropped = frame[int(y1):int(y2), int(x1):int(x2)]
        
# #         # MediaPipe processing
# #         results = pose.process(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
# #         if results.pose_landmarks:
# #             keypoints = []
# #             for landmark in results.pose_landmarks.landmark:
# #                 keypoints.append([
# #                     landmark.x, 
# #                     landmark.y, 
# #                     landmark.z
# #                 ])
            
# #             return keypoints
        
# #     return None

# # def main():
# #     cap = cv2.VideoCapture(0)
    
# #     # Countdown timer
# #     for i in range(5, 0, -1):
# #         ret, frame = cap.read()
# #         if ret:
# #             frame = draw_countdown(frame, i)
# #             cv2.imshow('Posture Analysis', frame)
# #             cv2.waitKey(1000)
    
# #     # Main loop
# #     while True:
# #         start_time = time.time()
# #         ret, frame = cap.read()
        
# #         if ret:
# #             # Process frame
# #             keypoints = process_frame(frame)
            
# #             if keypoints:
# #                 # Send to backend
# #                 response = requests.post(
# #                     API_ENDPOINT,
# #                     json={
# #                         "keypoints": keypoints,
# #                         "muscle_group": MUSCLE_GROUP,
# #                         "exercise": EXERCISE,
# #                         "timestamp": time.time()
# #                     }
# #                 )
                
# #                 if response.status_code == 200:
# #                     result = response.json()
# #                     # Display results
# #                     if 'class' in result:
# #                         label = "Correct" if result['class'] == 0 else "Incorrect"
# #                         cv2.putText(
# #                             frame,
# #                             f"{label} ({result['confidence']:.2f})",
# #                             (10, 30), 
# #                             cv2.FONT_HERSHEY_SIMPLEX, 
# #                             1, 
# #                             (0,255,0), 
# #                             2
# #                         )
                        
# #                         # Audio feedback
# #                         if result['class'] != 0:
# #                             # mixer.Sound('alert.wav').play()
# #                             mixer.Sound('assets/sounds/beep.wav').play()
# #                             print("Prediction Results: ", result['prediction'])
# #                             print("Prediction Class", result['class'])
            
# #             # Maintain target FPS
# #             processing_time = time.time() - start_time
# #             wait_time = max(1, int(1000/30 - processing_time*1000))
            
# #             cv2.imshow('Posture Analysis', frame)
# #             if cv2.waitKey(wait_time) & 0xFF == ord('q'):
# #                 break

# #     cap.release()
# #     cv2.destroyAllWindows()

# # if __name__ == "__main__":
# #     main()




# import cv2
# import time
# import requests
# import mediapipe as mp
# import torch
# import numpy as np
# import winsound
# import threading
# import uuid

# # Configuration
# MODEL_URL = "http://localhost:5000/predict"
# TARGET_FPS = 30
# MARGIN_RATIO = 0.2
# YOLO_INPUT_SIZE = 320
# SESSION_ID = str(uuid.uuid4())

# # Initialize models
# yolo = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
# yolo.conf = 0.5
# yolo.classes = [0]

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7)

# # User inputs (can be modified to GUI inputs)
# SELECTED_MUSCLE = "BACK"  # Replace with actual input
# SELECTED_EXERCISE = "deadlifts"  # Replace with actual input

# # Async request handler
# # class RequestHandler:
# #     def __init__(self):
# #         self.last_result = None
# #         self.last_update = time.time()
    
# #     def send_async(self, data):
# #         def worker():
# #             try:
# #                 response = requests.post(MODEL_URL, json=data, timeout=0.5)
# #                 if response.ok:
# #                     self.last_result = response.json()
# #                     self.last_update = time.time()
# #             except:
# #                 pass
# #         threading.Thread(target=worker).start()
    
# #     def get_result(self):
# #         if time.time() - self.last_update < 1.0:  # 1 second expiry
# #             return self.last_result
# #         return None

# # class RequestHandler:
# #     def __init__(self):
# #         self.last_result = {'status': 'initializing'}
# #         self.last_update = time.time()
    
# #     def send_async(self, data):
# #         def worker():
# #             try:
# #                 response = requests.post(MODEL_URL, json=data, timeout=0.5)
# #                 if response.ok:
# #                     self.last_result = response.json()
# #                     self.last_update = time.time()
# #                 else:
# #                     self.last_result = {'error': f'Server error: {response.status_code}'}
# #             except Exception as e:
# #                 self.last_result = {'error': f'Connection error: {str(e)}'}
# #         threading.Thread(target=worker).start()
    
# #     def get_result(self):
# #         if time.time() - self.last_update < 1.5:  # 1.5 second expiry
# #             return self.last_result
# #         return {'status': 'expired'}
# #         # return self.last_result


# # class RequestHandler:
# #     def __init__(self):
# #         self.last_result = {'status': 'ready'}
# #         self.last_update = time.time()
# #         self.request_in_progress = False
# #         self.lock = threading.Lock()
# #         self.cooldown = 0.5  # Minimum time between requests

# #     def send_async(self, data):
# #         if self.request_in_progress or (time.time() - self.last_update < self.cooldown):
# #             return
            
# #         def worker():
# #             try:
# #                 with self.lock:
# #                     self.request_in_progress = True
                
# #                 start_time = time.time()
# #                 response = requests.post(MODEL_URL, json=data, timeout=2.0)
                
# #                 with self.lock:
# #                     if response.ok:
# #                         self.last_result = response.json()
# #                         self.last_result['latency'] = time.time() - start_time
# #                         self.last_update = time.time()
# #                     else:
# #                         self.last_result = {
# #                             'error': f'Server error: {response.status_code}',
# #                             'status': 'error'
# #                         }
# #                     self.request_in_progress = False
                    
# #             except Exception as e:
# #                 with self.lock:
# #                     self.last_result = {
# #                         'error': f'Connection error: {str(e)}',
# #                         'status': 'error'
# #                     }
# #                     self.request_in_progress = False

# #         threading.Thread(target=worker).start()


# class RequestHandler:
#     def __init__(self):
#         self.last_result = {'status': 'ready'}
#         self.last_update = time.time()
#         self.request_in_progress = False
#         self.lock = threading.Lock()
#         self.cooldown = 1.0  # Increased cooldown

#     def send_async(self, data):
#         if self.request_in_progress or (time.time() - self.last_update < self.cooldown):
#             return
            
#         def worker():
#             try:
#                 with self.lock:
#                     self.request_in_progress = True
                
#                 start_time = time.time()
#                 response = requests.post(MODEL_URL, 
#                     json=data,
#                     timeout=3.0,
#                     headers={'Content-Type': 'application/json'}
#                 )
                
#                 with self.lock:
#                     if response.ok:
#                         self.last_result = response.json()
#                         self.last_update = time.time()
#                     else:
#                         error_msg = f"Server error: {response.status_code}"
#                         if response.status_code == 400:
#                             error_msg += f" - {response.text}"
#                         self.last_result = {'error': error_msg}
#                     self.request_in_progress = False
                    
#             except Exception as e:
#                 with self.lock:
#                     self.last_result = {'error': f"Connection error: {str(e)}"}
#                     self.request_in_progress = False

#         threading.Thread(target=worker).start()

#     def get_result(self):
#         with self.lock:
#             # Return last valid result for 2.5 seconds after update
#             if time.time() - self.last_update < 2.5:
#                 return self.last_result
#             # Show expired only if there was a previous valid result
#             if 'result' in self.last_result:
#                 return {'status': 'expired', 'message': 'Result expired - reposition and try again'}
#             return self.last_result


# handler = RequestHandler()

# def process_frame(frame):
#     # YOLO detection
#     results = yolo(frame, size=YOLO_INPUT_SIZE)
#     persons = results.xyxy[0][results.xyxy[0][:, 5] == 0]
    
#     if len(persons) > 0:
#         person = max(persons, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
#         x1, y1, x2, y2 = map(int, person[:4])
        
#         # Add padding
#         w, h = x2-x1, y2-y1
#         x1 = max(0, x1 - int(w*MARGIN_RATIO))
#         y1 = max(0, y1 - int(h*MARGIN_RATIO))
#         x2 = min(frame.shape[1], x2 + int(w*MARGIN_RATIO))
#         y2 = min(frame.shape[0], y2 + int(h*MARGIN_RATIO))
        
#         # MediaPipe processing
#         cropped = frame[y1:y2, x1:x2]
#         results = pose.process(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        
#         if results.pose_landmarks:
#             landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
#             return landmarks, cropped
    
#     return None, frame

# def main():
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
#     # Countdown
#     for i in range(5, 0, -1):
#         ret, frame = cap.read()
#         cv2.putText(frame, str(i), (300, 240), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
#         cv2.imshow('Posture Check', frame)
#         cv2.waitKey(1)
#         time.sleep(1)
    
#     last_frame_time = time.time()
#     fps_counter = []
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Process frame
#         start_time = time.time()
#         landmarks, processed_frame = process_frame(frame)
        
#         # Calculate FPS
#         fps_counter.append(time.time())
#         fps_counter = fps_counter[-30:]
#         real_fps = len(fps_counter) / (fps_counter[-1] - fps_counter[0]) if len(fps_counter) > 1 else 0
        
#         # if landmarks:
#         #     # Send async request
#         #     handler.send_async({
#         #         'session_id': SESSION_ID,
#         #         'keypoints': landmarks,
#         #         'muscle_group': SELECTED_MUSCLE,
#         #         'exercise': SELECTED_EXERCISE,
#         #         'fps': real_fps
#         #     })


#         if landmarks:
#         # Validate keypoints structure
#             if len(landmarks) != 33 or any(len(p) !=3 for p in landmarks):
#                 logging.error("Invalid landmarks detected")
#                 continue
                
#             # Send async request
#             handler.send_async({
#                 'session_id': SESSION_ID,
#                 'keypoints': landmarks,
#                 'muscle_group': SELECTED_MUSCLE,
#                 'exercise': SELECTED_EXERCISE,
#                 'fps': real_fps
#             })

        
#         # Get latest result
#         # result = handler.get_result()
#         # if result:
#         #     text = f"{result['result'].upper()} ({result.get('error', '')})"
#         #     color = (0, 255, 0) if result['result'] == 'correct' else (0, 0, 255)
#         #     cv2.putText(processed_frame, text, (10, 30),
#         #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         # result = handler.get_result()
#         # if result:
#         #     # Handle different response formats
#         #     if 'result' in result:
#         #         error_msg = result.get('error', '')
#         #         confidence = result.get('confidence', 0)
#         #         text = f"{result['result'].upper()} {error_msg} ({confidence:.0%})"
#         #         color = (0, 255, 0) if result['result'] == 'correct' else (0, 0, 255)
#         #     elif 'status' in result:
#         #         text = f"STATUS: {result['status'].upper()}"
#         #         color = (255, 255, 0)
#         #     elif 'error' in result:
#         #         text = f"ERROR: {result['error']}"
#         #         color = (0, 0, 255)
#         #     else:
#         #         text = "WAITING FOR ANALYSIS..."
#         #         color = (255, 255, 255)
    
#         #     cv2.putText(processed_frame, text, (10, 30),
#         #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         result = handler.get_result()
#         display_text = "Initializing..."
#         color = (255, 255, 255)  # White

#         if result.get('status') == 'expired':
#             display_text = result.get('message', 'Position expired')
#             color = (255, 255, 0)  # Yellow
#         elif 'result' in result:
#             error_msg = result.get('error', '')
#             confidence = result.get('confidence', 0)
#             display_text = f"{result['result'].upper()} {error_msg} ({confidence:.0%})"
#             color = (0, 255, 0) if result['result'] == 'correct' else (0, 0, 255)
#         elif 'error' in result:
#             display_text = f"ERROR: {result['error']}"
#             color = (0, 0, 255)  # Red

#         cv2.putText(processed_frame, display_text, (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
#         # Display FPS
#         cv2.putText(processed_frame, f"FPS: {real_fps:.1f}", (10, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         cv2.imshow('Posture Check', processed_frame)
        
#         # Maintain target FPS
#         elapsed = time.time() - start_time
#         delay = max(1, int((1/TARGET_FPS - elapsed)*1000))
#         if cv2.waitKey(delay) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()



import cv2
import torch
import numpy as np
import time
import requests
import threading
import queue
import pygame
from collections import deque
from playsound import playsound
import mediapipe as mp

# Configuration
BACKEND_URL = "http://localhost:8000/process-frame"
MUSCLE_GROUP = "BACK"  # Hardcoded
EXERCISE = "deadlifts"     # Hardcoded
FPS_TARGET = 30
BUFFER_SIZE = 5  # Number of frames to batch

# Initialize models
# yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# yolo.classes = [0]  # Person class only


yolo = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
yolo.conf = 0.5
yolo.classes = [0]


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Audio setup
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('assets/sounds/beep.wav')  # Provide beep.wav file

# Communication queue
send_queue = queue.Queue()
prediction_queue = deque(maxlen=10)

def countdown_timer():
    """Display countdown with audio"""
    for i in range(5, 0, -1):
        print(i)
        alert_sound.play()
        time.sleep(1)


def process_frame(frame):
    results = yolo(frame)
    
    if len(results.pred[0]) == 0:
        return None
    
    det = results.pred[0][results.pred[0][:, 4].argmax()]
    x1, y1, x2, y2 = det[:4].cpu().numpy()
    
    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - w * 0.2)
    y1 = max(0, y1 - h * 0.2)
    x2 = min(frame.shape[1], x2 + w * 0.2)
    y2 = min(frame.shape[0], y2 + h * 0.2)
    
    cropped = frame[int(y1):int(y2), int(x1):int(x2)]
    rgb_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_cropped)
    
    if results.pose_landmarks:
        # Draw landmarks on the cropped image
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            cropped,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )
    
    # Extract landmarks
    if not results.pose_landmarks:
        return None
    landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
    return landmarks


# def process_frame(frame):
#     results = yolo(frame)
    
#     # Handle YOLOv5n output format
#     if len(results.pred[0]) == 0:
#         return None
    
#     # Get best detection
#     det = results.pred[0][results.pred[0][:, 4].argmax()]
#     x1, y1, x2, y2 = det[:4].cpu().numpy()
    
#     # Expand bounding box (20% padding)
#     w, h = x2 - x1, y2 - y1
#     x1 = max(0, x1 - w * 0.2)
#     y1 = max(0, y1 - h * 0.2)
#     x2 = min(frame.shape[1], x2 + w * 0.2)
#     y2 = min(frame.shape[0], y2 + h * 0.2)
    
#     # Crop and process pose
#     cropped = frame[int(y1):int(y2), int(x1):int(x2)]
#     results = pose.process(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    
#     if not results.pose_landmarks:
#         return None
    
#     # Extract normalized landmarks
#     landmarks = []
#     for lmk in results.pose_landmarks.landmark:
#         landmarks.append([lmk.x, lmk.y, lmk.z])
    
#     return landmarks

def send_worker():
    """Batch process frames in background"""
    while True:
        batch = []
        while len(batch) < BUFFER_SIZE:
            item = send_queue.get()
            batch.append(item)
            if send_queue.empty():
                break
        
        try:
            response = requests.post(
                BACKEND_URL,
                json=[{
                    "landmarks": frame,
                    "muscle_group": MUSCLE_GROUP,
                    "exercise": EXERCISE,
                    "timestamp": time.time()
                } for frame in batch],
                timeout=0.5
            )
            if response.json().get('prediction'):
                prediction_queue.append(response.json()['prediction'])
        except Exception as e:
            print(f"API Error: {str(e)}")

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    
    # Start background worker
    threading.Thread(target=send_worker, daemon=True).start()
    
    # Countdown
    countdown_timer()
    
    # Frame timing control
    last_time = time.time()
    frame_interval = 1/FPS_TARGET
    
    while True:
        # Control frame rate
        current_time = time.time()
        if current_time - last_time < frame_interval:
            continue
        last_time = current_time
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        landmarks = process_frame(frame)
        if landmarks is None:
            continue
        
        # Add to processing queue
        send_queue.put(landmarks)

        # Display predictions

        if prediction_queue:
            prediction = prediction_queue[-1]
            status = f"Status: {prediction}"
            # Determine color based on prediction
            if 'incorrect' in prediction.lower():
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        
        # if prediction_queue:
        #     status = f"Status: {prediction_queue[-1]}"
        #     cv2.putText(frame, status, (10, 30),
        #                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Posture Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()