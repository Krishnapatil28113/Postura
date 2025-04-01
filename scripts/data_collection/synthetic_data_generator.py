# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# import json
# import csv
# import numpy as np
# from scipy.spatial.transform import Rotation
# from scripts.utils.normalization import hip_referenced_normalization, anthropometric_normalization

# # Joint mapping dictionary (extend as needed)
# JOINT_INDICES = {
#     # Face and Head
#     "nose": 0,
#     "left_eye_inner": 1,
#     "left_eye": 2,
#     "left_eye_outer": 3,
#     "right_eye_inner": 4,
#     "right_eye": 5,
#     "right_eye_outer": 6,
#     "left_ear": 7,
#     "right_ear": 8,
#     "mouth_left": 9,
#     "mouth_right": 10,
    
#     # Upper Body
#     "left_shoulder": 11,
#     "right_shoulder": 12,
#     "left_elbow": 13,
#     "right_elbow": 14,
#     "left_wrist": 15,
#     "right_wrist": 16,
    
#     # Hands
#     "left_pinky": 17,
#     "right_pinky": 18,
#     "left_index": 19,
#     "right_index": 20,
#     "left_thumb": 21,
#     "right_thumb": 22,
    
#     # Lower Body
#     "left_hip": 23,
#     "right_hip": 24,
#     "left_knee": 25,
#     "right_knee": 26,
#     "left_ankle": 27,
#     "right_ankle": 28,
    
#     # Feet
#     "left_heel": 29,
#     "right_heel": 30,
#     "left_foot_index": 31,
#     "right_foot_index": 32
# }

# def apply_transformation(keypoints, config):
#     transformed = keypoints.copy()
    
#     if config["transformation"] == "rotate":
#         angle = np.random.uniform(*config["angle_range"])
#         rot = Rotation.from_euler(config["axis"], angle, degrees=True)
#         for joint in config["affected_joints"]:
#             idx = JOINT_INDICES[joint]
#             transformed[:, idx] = rot.apply(transformed[:, idx])
            
#     elif config["transformation"] == "translate":
#         mag = np.random.uniform(*config["magnitude"])
#         direction = 1 if config["direction"] in ["forward", "upward", "right"] else -1
#         axis = 0 if config["direction"] in ["left", "right"] else 1
        
#         for joint in config["affected_joints"]:
#             idx = JOINT_INDICES[joint]
#             transformed[:, idx, axis] += direction * mag
            
#     return transformed



# def generate_synthetic_data(config_path, root_dir):
#     with open(config_path) as f:
#         config = json.load(f)
    
#     labels_path = os.path.join(root_dir, 'labels.csv')
#     label_exists = os.path.exists(labels_path)
    
#     with open(labels_path, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         if not label_exists:
#             writer.writerow(['filepath', 'muscle_group', 'exercise', 'is_correct', 'errors'])
        
#         # Iterate through muscle groups
#         for muscle_group in os.listdir(root_dir):
#             muscle_path = os.path.join(root_dir, muscle_group)
#             if not os.path.isdir(muscle_path):
#                 continue
            
#             # Iterate through exercises in muscle group
#             for exercise in os.listdir(muscle_path):
#                 exercise_path = os.path.join(muscle_path, exercise)
#                 if not os.path.isdir(exercise_path) or exercise not in config:
#                     continue
                
#                 # Process correct keypoints
#                 correct_dir = os.path.join(exercise_path, 'correct', 'keypoints')
#                 if os.path.exists(correct_dir):
#                     for kp_file in os.listdir(correct_dir):
#                         if kp_file.endswith('.npy'):
#                             rel_path = os.path.join(muscle_group, exercise, 'correct', 'keypoints', kp_file)
#                             writer.writerow([rel_path, muscle_group, exercise, 1, ''])
                
#                 # Generate incorrect keypoints
#                 incorrect_dir = os.path.join(exercise_path, 'incorrect', 'keypoints')
#                 os.makedirs(incorrect_dir, exist_ok=True)
                
#                 for error_name, params in config[exercise].items():
#                     error_slug = error_name.lower().replace(' ', '_')
                    
#                     # Get correct samples
#                     correct_dir = os.path.join(exercise_path, 'correct', 'keypoints')
#                     if not os.path.exists(correct_dir):
#                         continue
                        
#                     for kp_file in os.listdir(correct_dir):
#                         if not kp_file.endswith('.npy'):
#                             continue
                            
#                         # Load and transform
#                         correct_kps = np.load(os.path.join(correct_dir, kp_file))
#                         transformed_kps = apply_transformation(correct_kps, params)
                        
#                         # Normalize
#                         transformed_kps = hip_referenced_normalization(transformed_kps)
#                         transformed_kps = anthropometric_normalization(transformed_kps)
                        
#                         # Save
#                         base_name = os.path.splitext(kp_file)[0]
#                         new_filename = f"{base_name}_{error_slug}.npy"
#                         save_path = os.path.join(incorrect_dir, new_filename)
#                         np.save(save_path, transformed_kps)
                        
#                         # Write to CSV
#                         rel_path = os.path.join(muscle_group, exercise, 'incorrect', 'keypoints', new_filename)
#                         writer.writerow([rel_path, muscle_group, exercise, 0, error_slug])

# if __name__ == "__main__":
#     generate_synthetic_data(
#         "form_errors_config.json",
#         "assets/datasets/raw"
#     )





# ------------------------------------------------------------------------------------


import os
import sys
import json
import csv
import numpy as np
from scipy.spatial.transform import Rotation

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# from scripts.utils.normalization import hip_referenced_normalization, anthropometric_normalization



# Joint mapping dictionary (extend as needed)
JOINT_INDICES = {
    # Face and Head
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    
    # Upper Body
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    
    # Hands
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    
    # Lower Body
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    
    # Feet
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32
}

def is_valid_keypoint(kp):
    """Check if a keypoint is valid (not NaN)"""
    return not np.any(np.isnan(kp))

def apply_transformation(keypoints, config):
    transformed = keypoints.copy()
    trans_type = config["transformation"]
    
    # Helper function to get joint index
    def get_joint_idx(joint_name):
        return JOINT_INDICES[joint_name]

    # Rotation transformation
    if trans_type == "rotate":
        angle = np.random.uniform(*config["angle_range"])
        axis = config["axis"]
        rot = Rotation.from_euler(axis, angle, degrees=True)
        
        for joint in config["affected_joints"]:
            idx = get_joint_idx(joint)
            for i in range(transformed.shape[0]):
                if is_valid_keypoint(transformed[i, idx]):
                    transformed[i, idx] = rot.apply(transformed[i, idx])

    # Translation transformation
    elif trans_type == "translate":
        mag = np.random.uniform(*config["magnitude"])
        direction = config["direction"]
        axis_mapping = {
            "forward": 0, "backward": 0,
            "left": 1, "right": 1,
            "upward": 2, "downward": 2
        }
        axis = axis_mapping[direction]
        sign = 1 if direction in ["forward", "right", "upward"] else -1

        for joint in config["affected_joints"]:
            idx = get_joint_idx(joint)
            for i in range(transformed.shape[0]):
                if is_valid_keypoint(transformed[i, idx]):
                    transformed[i, idx, axis] += sign * mag

    # Vertical wave pattern (for bar path deviations)
    elif trans_type == "vertical_wave":
        freq = np.random.uniform(*config["frequency"])
        amp = np.random.uniform(*config["amplitude"])
        frames = transformed.shape[0]
        wave = amp * np.sin(2 * np.pi * freq * np.linspace(0, 1, frames))
        
        for joint in config["affected_joints"]:
            idx = get_joint_idx(joint)
            for i in range(frames):
                if is_valid_keypoint(transformed[i, idx]):
                    transformed[i, idx, 1] += wave[i]  # Y-axis

    # Asymmetric translation (different magnitudes for left/right)
    elif trans_type == "asymmetric_translate":
        left_mag = np.random.uniform(*config["left_magnitude"])
        right_mag = np.random.uniform(*config["right_magnitude"])
        axis = 0 if config.get("direction", "forward") in ["forward", "backward"] else 1

        for joint in config["affected_joints"]:
            idx = get_joint_idx(joint)
            mag = left_mag if "left" in joint else right_mag
            for i in range(transformed.shape[0]):
                if is_valid_keypoint(transformed[i, idx]):
                    transformed[i, idx, axis] += mag

    # Temporal shift (progressive offset over frames)
    elif trans_type == "temporal_shift":
        shift_frames = np.random.randint(*config["shift_frames"])
        mag = np.random.uniform(*config["magnitude"])
        axis = 0  # X-axis for forward/backward movement

        for joint in config["affected_joints"]:
            idx = get_joint_idx(joint)
            for i in range(transformed.shape[0]):
                if is_valid_keypoint(transformed[i, idx]):
                    offset = mag * (i/shift_frames if i < shift_frames else 1)
                    transformed[i, idx, axis] += offset

    # Range limitation (angle constraints)
    elif trans_type == "limit_range":
        min_ang = np.radians(config["min_angle"])
        max_ang = np.radians(config["max_angle"])
        
        for joint in config["affected_joints"]:
            idx = get_joint_idx(joint)
            for i in range(transformed.shape[0]):
                if is_valid_keypoint(transformed[i, idx]):
                    # Calculate current angle
                    vec = transformed[i, idx] - transformed[i, JOINT_INDICES["left_hip" if "left" in joint else "right_hip"]]
                    angle = np.arctan2(vec[2], vec[0])  # Z-X plane angle
                    
                    # Apply constraints
                    if angle < min_ang:
                        ratio = (min_ang - angle)/(angle + 1e-8)
                        transformed[i, idx] *= (1 + ratio)
                    elif angle > max_ang:
                        ratio = (angle - max_ang)/(angle + 1e-8)
                        transformed[i, idx] *= (1 - ratio)

    # Add custom transformations here
    else:
        raise ValueError(f"Unknown transformation type: {trans_type}")

    return transformed

def generate_synthetic_data(config_path, root_dir):
    with open(config_path) as f:
        config = json.load(f)
    
    labels_path = os.path.join(root_dir, 'labels.csv')
    label_exists = os.path.exists(labels_path)
    
    with open(labels_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not label_exists:
            writer.writerow(['filepath', 'muscle_group', 'exercise', 'is_correct', 'errors'])
        
        # Iterate through muscle groups
        for muscle_group in os.listdir(root_dir):
            muscle_path = os.path.join(root_dir, muscle_group)
            if not os.path.isdir(muscle_path):
                continue
            
            # Iterate through exercises in muscle group
            for exercise in os.listdir(muscle_path):
                exercise_path = os.path.join(muscle_path, exercise)
                if not os.path.isdir(exercise_path) or exercise not in config:
                    continue
                
                # Process correct keypoints
                correct_dir = os.path.join(exercise_path, 'correct', 'keypoints')
                if os.path.exists(correct_dir):
                    for kp_file in os.listdir(correct_dir):
                        if kp_file.endswith('.npy'):
                            rel_path = os.path.join(muscle_group, exercise, 'correct', 'keypoints', kp_file)
                            writer.writerow([rel_path, muscle_group, exercise, 1, ''])
                
                # Generate incorrect keypoints
                incorrect_dir = os.path.join(exercise_path, 'incorrect', 'keypoints')
                os.makedirs(incorrect_dir, exist_ok=True)
                
                for error_name, params in config[exercise].items():
                    error_slug = error_name.lower().replace(' ', '_')
                    
                    # Get correct samples
                    correct_dir = os.path.join(exercise_path, 'correct', 'keypoints')
                    if not os.path.exists(correct_dir):
                        continue
                        
                    for kp_file in os.listdir(correct_dir):
                        if not kp_file.endswith('.npy'):
                            continue
                            
                        # Load and transform
                        correct_kps = np.load(os.path.join(correct_dir, kp_file))
                        transformed_kps = apply_transformation(correct_kps, params)
                        
                        # Normalize
                        # transformed_kps = hip_referenced_normalization(transformed_kps)
                        # transformed_kps = anthropometric_normalization(transformed_kps)
                        
                        # Save
                        base_name = os.path.splitext(kp_file)[0]
                        new_filename = f"{base_name}_{error_slug}.npy"
                        save_path = os.path.join(incorrect_dir, new_filename)
                        np.save(save_path, transformed_kps)
                        
                        # Write to CSV
                        rel_path = os.path.join(muscle_group, exercise, 'incorrect', 'keypoints', new_filename)
                        writer.writerow([rel_path, muscle_group, exercise, 0, error_slug])

if __name__ == "__main__":
    generate_synthetic_data(
        "form_errors_config.json",
        "assets/datasets/raw"
    )