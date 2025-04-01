# import numpy as np
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from tensorflow.keras.models import load_model
# import json
# import joblib
# from scipy import signal
# import logging
# from datetime import datetime

# import sys
# import os

# # Add root directory to Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# app = FastAPI()

# # Load trained model and encoders
# model = load_model('assets\models\posture_model_final.keras')
# muscle_encoder = joblib.load('assets\models\muscle_encoder.joblib')
# exercise_encoder = joblib.load('assets\models\exercise_encoder.joblib')

# # Configuration
# TARGET_FPS = 30
# SEQUENCE_LENGTH = 150  # 5 seconds * 30 fps
# buffer = []
# last_processed = datetime.now()

# class FrameData(BaseModel):
#     keypoints: list  # 33 landmarks with x,y,z coordinates
#     muscle_group: str
#     exercise: str
#     timestamp: float

# def resample_to_target_fps(frames, original_fps):
#     """Resample frames to target FPS using linear interpolation"""
#     num_frames = len(frames)
#     x_old = np.linspace(0, 1, num_frames)
#     x_new = np.linspace(0, 1, int(num_frames * TARGET_FPS / original_fps))
    
#     resampled = []
#     for joint in range(33):
#         for coord in range(3):
#             resampled_coord = np.interp(x_new, x_old, [f[joint][coord] for f in frames])
#             resampled.append(resampled_coord)
    
#     return np.reshape(resampled, (-1, 33, 3)).tolist()

# def preprocess_input(frames):
#     """Match training preprocessing steps"""
#     # Convert to numpy array
#     arr = np.array(frames)
    
#     # Normalize per sample
#     mean = np.mean(arr, axis=(0, 1))
#     std = np.std(arr, axis=(0, 1)) + 1e-8
#     arr = (arr - mean) / std
    
#     # Pad or truncate to sequence length
#     if len(arr) < SEQUENCE_LENGTH:
#         pad = SEQUENCE_LENGTH - len(arr)
#         arr = np.pad(arr, ((0,pad),(0,0),(0,0)), mode='edge')
#     else:
#         arr = arr[:SEQUENCE_LENGTH]
    
#     return np.expand_dims(arr, 0)

# @app.post("/process-frame")
# async def process_frame(frame_data: FrameData):
#     global buffer, last_processed
    
#     try:
#         # Add to buffer with timestamp
#         buffer.append({
#             'keypoints': frame_data.keypoints,
#             'muscle_group': frame_data.muscle_group,
#             'exercise': frame_data.exercise,
#             'timestamp': frame_data.timestamp
#         })
        
#         # Calculate current FPS
#         time_diff = (datetime.now() - last_processed).total_seconds()
#         current_fps = len(buffer) / time_diff if time_diff > 0 else 0
        
#         # Process when buffer has enough data
#         if len(buffer) >= SEQUENCE_LENGTH:
#             # Resample if needed
#             if abs(current_fps - TARGET_FPS) > 5:
#                 resampled = resample_to_target_fps([f['keypoints'] for f in buffer], current_fps)
#                 processed_frames = resampled[:SEQUENCE_LENGTH]
#             else:
#                 processed_frames = [f['keypoints'] for f in buffer][:SEQUENCE_LENGTH]
            
#             # Encode categorical features
#             muscle_encoded = muscle_encoder.transform(buffer[0]['muscle_group'])[0]
#             exercise_encoded = exercise_encoder.transform(buffer[0]['exercise'])[0]
            
#             # Preprocess data
#             processed_data = preprocess_input(processed_frames)
            
#             # Make prediction
#             prediction = model.predict([
#                 processed_data,
#                 np.array([[muscle_encoded]]),
#                 np.array([[exercise_encoded]])
#             ])
            
#             # Clear buffer
#             buffer = []
#             last_processed = datetime.now()
            
#             return {
#                 "prediction": prediction[0].tolist(),
#                 "class": int(np.argmax(prediction)),
#                 "confidence": float(np.max(prediction))
#             }
            
#         return {"status": "buffering"}

#     except Exception as e:
#         logging.error(f"Processing error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)



import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib
from scipy.interpolate import interp1d
import threading

app = Flask(__name__)


import os
from pathlib import Path

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent  # Points to E:\postura

# Update model paths
model_path = BASE_DIR / 'assets' / 'models' / 'posture_model_final.keras'
muscle_encoder_path = BASE_DIR / 'assets' / 'models' / 'muscle_encoder.joblib'
exercise_encoder_path = BASE_DIR / 'assets' / 'models' / 'exercise_encoder.joblib'

# Load assets with verification
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = load_model(model_path)
muscle_encoder = joblib.load(muscle_encoder_path)
exercise_encoder = joblib.load(exercise_encoder_path)


# Load assets
# model_path = os.path.join(os.path.dirname(__file__), 'assets\models\posture_model_final.keras')
# model = load_model(model_path)
# muscle_encoder = joblib.load('assets\models\muscle_encoder.joblib')
# exercise_encoder = joblib.load('assets\models\exercise_encoder.joblib')

# Buffer for temporal data
BUFFER_SIZE = 150  # 5 seconds at 30fps
frame_buffer = []
buffer_lock = threading.Lock()

def resample_frames(frames, original_fps, target_fps=30):
    """Resample frames to target FPS using linear interpolation"""
    original_time = np.linspace(0, 1, len(frames))
    target_time = np.linspace(0, 1, int(len(frames) * target_fps / original_fps))
    
    interpolator = interp1d(original_time, frames, axis=0, kind='linear')
    resampled = interpolator(target_time)
    return resampled[:BUFFER_SIZE]

def preprocess_data(keypoints, muscle_group, exercise, fps):
    with buffer_lock:
        # Add new frame to buffer
        frame_buffer.append(np.array(keypoints))
        
        # Maintain buffer size according to detected FPS
        if len(frame_buffer) > BUFFER_SIZE * (fps / 30):
            frame_buffer.pop(0)
            
        # Resample if needed
        if abs(fps - 30) > 2:
            resampled = resample_frames(frame_buffer, fps)
        else:
            resampled = np.array(frame_buffer[-BUFFER_SIZE:])
        
        # Pad if insufficient frames
        if len(resampled) < BUFFER_SIZE:
            resampled = np.pad(resampled, ((0, BUFFER_SIZE-len(resampled)), (0,0), (0,0)), 
                           mode='edge')
        
        # Normalize
        mean = resampled.mean(axis=(0, 1))
        std = resampled.std(axis=(0, 1)) + 1e-8
        normalized = (resampled - mean) / std
        
        # Prepare other inputs
        muscle_encoded = muscle_encoder.transform([muscle_group])[0]
        exercise_encoded = exercise_encoder.transform([exercise])[0]
        
        return (
            normalized[np.newaxis, ...],  # Add batch dimension
            np.array([muscle_encoded]),
            np.array([exercise_encoded])
        )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Get input data
        keypoints = data['keypoints']
        muscle_group = data['muscle_group']
        exercise = data['exercise']
        fps = data.get('fps', 30)
        
        # Preprocess
        processed_data = preprocess_data(keypoints, muscle_group, exercise, fps)
        
        # Predict
        prediction = model.predict(processed_data)
        class_idx = np.argmax(prediction)
        
        # Get class label
        if class_idx == 0:
            return jsonify({'result': 'correct', 'confidence': float(prediction[0][class_idx])})
        else:
            return jsonify({
                'result': 'incorrect',
                'error': muscle_encoder.inverse_transform([class_idx])[0],
                'confidence': float(prediction[0][class_idx])
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)