import os

MUSCLE_GROUPS = {
    "CHEST": ["bench_press", "push_ups"],
    "SHOULDER": ["overhead_press", "lateral_raises"],
    "BACK": ["pull_ups", "deadlifts"],
    "LEGS": ["squats", "leg_extension"]
}

def create_dataset_structure(base_path):
    print(f"Creating dataset structure at: {base_path}")
    for muscle, exercises in MUSCLE_GROUPS.items():
        muscle_path = os.path.join(base_path, muscle)
        os.makedirs(muscle_path, exist_ok=True)
        
        for exercise in exercises:
            exercise_path = os.path.join(muscle_path, exercise)

            correct_path = os.path.join(exercise_path, "correct")
            os.makedirs(correct_path, exist_ok=True)

            correct_video_path = os.path.join(correct_path, "video")
            os.makedirs(correct_video_path, exist_ok=True)

            correct_keypoints_path = os.path.join(correct_path, "keypoints")
            os.makedirs(correct_keypoints_path, exist_ok=True)
            
            incorrect_path = os.path.join(exercise_path, "incorrect")
            os.makedirs(incorrect_path, exist_ok=True)

            incorrect_keypoints_path = os.path.join(incorrect_path, "keypoints")
            os.makedirs(incorrect_keypoints_path, exist_ok=True)

create_dataset_structure("assets/datasets/raw")