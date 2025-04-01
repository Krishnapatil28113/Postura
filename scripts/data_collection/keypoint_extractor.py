import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
# from scripts.utils.normalization import hip_referenced_normalization, anthropometric_normalization

mp_pose = mp.solutions.pose

def process_video(video_path):
    """Extract and normalize keypoints from a video"""
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    
    with mp_pose.Pose(static_image_mode=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                frame_kps = np.array([[lmk.x, lmk.y, lmk.z] 
                      for lmk in results.pose_landmarks.landmark])
                keypoints.append(frame_kps)
    
    cap.release()
    if not keypoints:
        return None
    
    # Convert to numpy array [num_frames, 33, 3]
    keypoints = np.array(keypoints)
    
    # Apply normalization
    # keypoints = hip_referenced_normalization(keypoints)
    # keypoints = anthropometric_normalization(keypoints)
    
    return keypoints

def process_all_videos(root_dir):

    print(f"Root directory: {os.path.abspath(root_dir)}")
    """Process all videos in the dataset"""
    for muscle in os.listdir(root_dir):

        muscle_path = os.path.join(root_dir, muscle)
        print(f"Processing muscle: {muscle_path}")

        for exercise in os.listdir(muscle_path):
            exercise_path = os.path.join(muscle_path, exercise)
            # correct_dir = os.path.join(exercise_path, "correct")
            video_dir = os.path.join(exercise_path, "correct", "video")

            print(f"  Looking for videos in: {video_dir}")

            if not os.path.exists(video_dir):
                print(f"  ❌ Folder does not exist: {video_dir}")
                continue
            

            files = os.listdir(video_dir)
            print(f"  Files found: {files}")
            # video_dir = os.path.join(correct_dir, "video")
            output_dir = os.path.join(exercise_path, "correct", "keypoints")
            
            flag = False
            for video_file in tqdm(os.listdir(video_dir), desc=f"{muscle}/{exercise}"):
                flag = True
                if not video_file.lower().endswith((".mp4", ".mov", ".avi")):
                    continue
                
                video_path = os.path.join(video_dir, video_file)
                keypoints = process_video(video_path)
                
                if keypoints is not None:
                    base_name = os.path.splitext(video_file)[0]
                    np.save(os.path.join(output_dir, f"{base_name}.npy"), keypoints)
                else:
                    print("Keypoints are NONE")
            
            if flag == False:
                print("No Video getting processed")
            else:
                print("Everything All right")
if __name__ == "__main__":
    process_all_videos("assets/datasets/raw")





# -------------------------------------------------------------

# import sys
# import os
# import cv2
# import numpy as np
# import mediapipe as mp
# from tqdm import tqdm

# # Add project root to Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# from scripts.utils.normalization import hip_referenced_normalization, anthropometric_normalization


# mp_pose = mp.solutions.pose

# VISIBILITY_THRESHOLD = 0.6  # Minimum visibility score to consider a keypoint valid

# def process_video(video_path):
#     """Extract and normalize keypoints from a video"""
#     cap = cv2.VideoCapture(video_path)
#     keypoints = []
    
#     with mp_pose.Pose(static_image_mode=False) as pose:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             # Convert BGR to RGB and process
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(frame_rgb)
            
#             if results.pose_landmarks:
#                 # Process keypoints with visibility check
#                 frame_kps = []
#                 for lmk in results.pose_landmarks.landmark:
#                     if lmk.visibility < VISIBILITY_THRESHOLD:
#                         # Set coordinates to NaN if visibility is low
#                         frame_kps.append([np.nan, np.nan, np.nan])
#                     else:
#                         frame_kps.append([lmk.x, lmk.y, lmk.z])
                
#                 keypoints.append(np.array(frame_kps))
    
#     cap.release()
#     if not keypoints:
#         return None
    
#     # Convert to numpy array [num_frames, 33, 3]
#     keypoints = np.array(keypoints)
    
#     # Apply normalization
#     # keypoints = hip_referenced_normalization(keypoints)
#     # keypoints = anthropometric_normalization(keypoints)
    
#     return keypoints

# def process_all_videos(root_dir):
#     """Process all videos in the dataset"""
#     print(f"Root directory: {os.path.abspath(root_dir)}")
    
#     for muscle in os.listdir(root_dir):
#         muscle_path = os.path.join(root_dir, muscle)
#         print(f"Processing muscle: {muscle_path}")

#         for exercise in os.listdir(muscle_path):
#             exercise_path = os.path.join(muscle_path, exercise)
#             video_dir = os.path.join(exercise_path, "correct", "video")
#             output_dir = os.path.join(exercise_path, "correct", "keypoints")

#             print(f"  Looking for videos in: {video_dir}")
#             if not os.path.exists(video_dir):
#                 print(f"  ❌ Folder does not exist: {video_dir}")
#                 continue

#             # Create output directory if it doesn't exist
#             os.makedirs(output_dir, exist_ok=True)

#             files = os.listdir(video_dir)
#             print(f"  Files found: {files}")
            
#             processed_flag = False
#             for video_file in tqdm(os.listdir(video_dir), desc=f"{muscle}/{exercise}"):
#                 if not video_file.lower().endswith((".mp4", ".mov", ".avi")):
#                     continue
                
#                 processed_flag = True
#                 video_path = os.path.join(video_dir, video_file)
#                 keypoints = process_video(video_path)
                
#                 if keypoints is not None:
#                     base_name = os.path.splitext(video_file)[0]
#                     np.save(os.path.join(output_dir, f"{base_name}.npy"), keypoints)
#                 else:
#                     print(f"⚠️ No keypoints detected in {video_file}")

#             if not processed_flag:
#                 print("  No valid video files processed")
#             else:
#                 print("  Processing completed successfully")

# if __name__ == "__main__":
#     process_all_videos("assets/datasets/raw")