# # import os
# # import pandas as pd
# # import numpy as np

# # MAX_FRAMES = 120
# # labels_path = os.path.join('assets', 'datasets', 'raw', 'labels.csv')
# # dataset_root = os.path.join('assets', 'datasets', 'raw')



# # # Load dataset
# # df = pd.read_csv(labels_path)
# # df['filepath'] = df['filepath'].apply(lambda x: os.path.join(
# #     dataset_root,
# #     x.replace('\\', '/')  # Convert Windows paths to Unix-style
# # ))

# # for path in df['filepath']:
# #     print(path)


# # X_kp = []
# # for path in df['filepath']:
# #     path = path.replace('\\', '/').strip()
# #     path = dataset_root + '/' + path
# #     print(path)
# #     try:
# #         kp = np.load(path)
# #     except FileNotFoundError:
# #         print(f"File not found: {path}")
# #         continue
        
# #     # Pad/truncate to MAX_FRAMES
# #     if kp.shape[0] < MAX_FRAMES:
# #         pad = np.zeros((MAX_FRAMES - kp.shape[0], *kp.shape[1:]))
# #         kp = np.vstack([kp, pad])
# #     else:
# #         kp = kp[:MAX_FRAMES]
# #     X_kp.append(kp)

# # X_kp = np.array(X_kp)




# # # Add these diagnostics after loading the data
# # print(f"Total samples: {len(df)}")
# # print("Class distribution:")
# # print(df['is_correct'].value_counts())
# # print("Sample keypoints shape:", X_kp[0].shape)
# # print("Keypoints min/max:", X_kp[0].min(), X_kp[0].max())

# # # Verify first 5 paths
# # for path in df['filepath'].head():
# #     print("Exists:", os.path.exists(path), "| Path:", path)



# import cv2
# import mediapipe as mp

# # Initialize MediaPipe drawing utilities
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# def visualize_keypoints(video_path):
#     """Show real-time pose estimation visualization"""
#     cap = cv2.VideoCapture(video_path)
    
#     with mp_pose.Pose(
#         static_image_mode=False,
#         model_complexity=2,  # Highest accuracy
#         enable_segmentation=False,
#         min_detection_confidence=0.7
#     ) as pose:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Convert BGR to RGB and process
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(frame_rgb)

#             # Draw pose landmarks on the frame
#             annotated_frame = frame.copy()
#             if results.pose_landmarks:
#                 mp_drawing.draw_landmarks(
#                     annotated_frame,
#                     results.pose_landmarks,
#                     mp_pose.POSE_CONNECTIONS,
#                     mp_drawing.DrawingSpec(color=(0,255,0)),  # Joint color
#                     mp_drawing.DrawingSpec(color=(255,0,0)))  # Connection color

#             # Display the annotated frame
#             cv2.imshow('Pose Detection', annotated_frame)
            
#             # Exit on 'q' press
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Use 0 for webcam or provide video path
#     visualize_keypoints("assets/datasets/raw/BACK/pull_ups/correct/video/pull up_1.mp4")  # Replace with your video path


import cv2
import numpy as np
import requests
import time
import threading
from playsound import playsound

# Configuration
BACKEND_URL = "http://localhost:5000/api/realtime"  # Update with your backend URL
EXERCISE_DURATION = 30  # seconds
HARDCODED_MUSCLE_GROUP = "BACK"
HARDCODED_EXERCISE = "deadlift"

def countdown_timer():
    """Display countdown with audio feedback"""
    for i in range(5, 0, -1):
        print(f"Starting in {i}...")
        # Display on camera frame
        display_text(f"Starting in {i}...")
        # Play sound in background
        threading.Thread(target=playsound, args=('beep.wav',), daemon=True).start()
        time.sleep(1)

def display_text(text, frame):
    """Helper function to display text on frame"""
    cv2.putText(frame, text, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

def process_frame(frame):
    """Send frame to backend API and get response"""
    # Convert frame to JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)
    
    try:
        response = requests.post(
            BACKEND_URL,
            files={
                'frame': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg'),
            },
            data={
                'muscle_group': HARDCODED_MUSCLE_GROUP,
                'exercise': HARDCODED_EXERCISE,
                'fps': 30
            },
            stream=True,
            timeout=0.5
        )
        
        if response.status_code == 200:
            return response.json()
            
    except Exception as e:
        print(f"API Error: {str(e)}")
    
    return None

def realtime_analysis():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Show initial frame with instructions
    ret, frame = cap.read()
    display_text("Get ready for exercise!", frame)
    cv2.imshow("Exercise Analysis", frame)
    cv2.waitKey(2000)
    
    # Run countdown
    countdown_timer()
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Mirror frame
        frame = cv2.flip(frame, 1)
        
        # Process frame through API
        feedback = process_frame(frame)
        
        # Display feedback
        if feedback:
            display_text(f"Status: {feedback.get('status', '')}", frame)
            display_text(f"Feedback: {feedback.get('message', '')}", frame, y=100)
        
        # Show timer
        elapsed = time.time() - start_time
        display_text(f"Time: {int(elapsed)}s / {EXERCISE_DURATION}s", frame, y=150)
        
        cv2.imshow("Exercise Analysis", frame)
        
        # Exit conditions
        if cv2.waitKey(1) & 0xFF == ord('q') or elapsed > EXERCISE_DURATION:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Verify webcam access
    test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        print("Error: Webcam not accessible")
    else:
        test_cap.release()
        realtime_analysis()