# # # import numpy as np
# # # from fastapi import FastAPI, HTTPException
# # # from pydantic import BaseModel
# # # from tensorflow.keras.models import load_model
# # # import json
# # # import joblib
# # # from scipy import signal
# # # import logging
# # # from datetime import datetime

# # # import sys
# # # import os

# # # # Add root directory to Python path
# # # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # # app = FastAPI()

# # # # Load trained model and encoders
# # # model = load_model('assets\models\posture_model_final.keras')
# # # muscle_encoder = joblib.load('assets\models\muscle_encoder.joblib')
# # # exercise_encoder = joblib.load('assets\models\exercise_encoder.joblib')

# # # # Configuration
# # # TARGET_FPS = 30
# # # SEQUENCE_LENGTH = 150  # 5 seconds * 30 fps
# # # buffer = []
# # # last_processed = datetime.now()

# # # class FrameData(BaseModel):
# # #     keypoints: list  # 33 landmarks with x,y,z coordinates
# # #     muscle_group: str
# # #     exercise: str
# # #     timestamp: float

# # # def resample_to_target_fps(frames, original_fps):
# # #     """Resample frames to target FPS using linear interpolation"""
# # #     num_frames = len(frames)
# # #     x_old = np.linspace(0, 1, num_frames)
# # #     x_new = np.linspace(0, 1, int(num_frames * TARGET_FPS / original_fps))
    
# # #     resampled = []
# # #     for joint in range(33):
# # #         for coord in range(3):
# # #             resampled_coord = np.interp(x_new, x_old, [f[joint][coord] for f in frames])
# # #             resampled.append(resampled_coord)
    
# # #     return np.reshape(resampled, (-1, 33, 3)).tolist()

# # # def preprocess_input(frames):
# # #     """Match training preprocessing steps"""
# # #     # Convert to numpy array
# # #     arr = np.array(frames)
    
# # #     # Normalize per sample
# # #     mean = np.mean(arr, axis=(0, 1))
# # #     std = np.std(arr, axis=(0, 1)) + 1e-8
# # #     arr = (arr - mean) / std
    
# # #     # Pad or truncate to sequence length
# # #     if len(arr) < SEQUENCE_LENGTH:
# # #         pad = SEQUENCE_LENGTH - len(arr)
# # #         arr = np.pad(arr, ((0,pad),(0,0),(0,0)), mode='edge')
# # #     else:
# # #         arr = arr[:SEQUENCE_LENGTH]
    
# # #     return np.expand_dims(arr, 0)

# # # @app.post("/process-frame")
# # # async def process_frame(frame_data: FrameData):
# # #     global buffer, last_processed
    
# # #     try:
# # #         # Add to buffer with timestamp
# # #         buffer.append({
# # #             'keypoints': frame_data.keypoints,
# # #             'muscle_group': frame_data.muscle_group,
# # #             'exercise': frame_data.exercise,
# # #             'timestamp': frame_data.timestamp
# # #         })
        
# # #         # Calculate current FPS
# # #         time_diff = (datetime.now() - last_processed).total_seconds()
# # #         current_fps = len(buffer) / time_diff if time_diff > 0 else 0
        
# # #         # Process when buffer has enough data
# # #         if len(buffer) >= SEQUENCE_LENGTH:
# # #             # Resample if needed
# # #             if abs(current_fps - TARGET_FPS) > 5:
# # #                 resampled = resample_to_target_fps([f['keypoints'] for f in buffer], current_fps)
# # #                 processed_frames = resampled[:SEQUENCE_LENGTH]
# # #             else:
# # #                 processed_frames = [f['keypoints'] for f in buffer][:SEQUENCE_LENGTH]
            
# # #             # Encode categorical features
# # #             muscle_encoded = muscle_encoder.transform(buffer[0]['muscle_group'])[0]
# # #             exercise_encoded = exercise_encoder.transform(buffer[0]['exercise'])[0]
            
# # #             # Preprocess data
# # #             processed_data = preprocess_input(processed_frames)
            
# # #             # Make prediction
# # #             prediction = model.predict([
# # #                 processed_data,
# # #                 np.array([[muscle_encoded]]),
# # #                 np.array([[exercise_encoded]])
# # #             ])
            
# # #             # Clear buffer
# # #             buffer = []
# # #             last_processed = datetime.now()
            
# # #             return {
# # #                 "prediction": prediction[0].tolist(),
# # #                 "class": int(np.argmax(prediction)),
# # #                 "confidence": float(np.max(prediction))
# # #             }
            
# # #         return {"status": "buffering"}

# # #     except Exception as e:
# # #         logging.error(f"Processing error: {str(e)}")
# # #         raise HTTPException(status_code=500, detail=str(e))

# # # if __name__ == "__main__":
# # #     import uvicorn
# # #     uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)



# # import numpy as np
# # from flask import Flask, request, jsonify
# # from tensorflow.keras.models import load_model
# # from sklearn.preprocessing import LabelEncoder
# # import joblib
# # from scipy.interpolate import interp1d
# # import threading

# # app = Flask(__name__)


# # import os
# # from pathlib import Path

# # # Get the base directory of the project
# # BASE_DIR = Path(__file__).resolve().parent.parent  # Points to E:\postura

# # # Update model paths
# # model_path = BASE_DIR / 'assets' / 'models' / 'posture_model_final.keras'
# # muscle_encoder_path = BASE_DIR / 'assets' / 'models' / 'muscle_encoder.joblib'
# # exercise_encoder_path = BASE_DIR / 'assets' / 'models' / 'exercise_encoder.joblib'

# # # Load assets with verification
# # if not model_path.exists():
# #     raise FileNotFoundError(f"Model file not found at {model_path}")

# # model = load_model(model_path)
# # muscle_encoder = joblib.load(muscle_encoder_path)
# # exercise_encoder = joblib.load(exercise_encoder_path)


# # # Load assets
# # # model_path = os.path.join(os.path.dirname(__file__), 'assets\models\posture_model_final.keras')
# # # model = load_model(model_path)
# # # muscle_encoder = joblib.load('assets\models\muscle_encoder.joblib')
# # # exercise_encoder = joblib.load('assets\models\exercise_encoder.joblib')

# # # Buffer for temporal data
# # BUFFER_SIZE = 150  # 5 seconds at 30fps
# # frame_buffer = []
# # buffer_lock = threading.Lock()

# # def resample_frames(frames, original_fps, target_fps=30):
# #     """Resample frames to target FPS using linear interpolation"""
# #     original_time = np.linspace(0, 1, len(frames))
# #     target_time = np.linspace(0, 1, int(len(frames) * target_fps / original_fps))
    
# #     interpolator = interp1d(original_time, frames, axis=0, kind='linear')
# #     resampled = interpolator(target_time)
# #     return resampled[:BUFFER_SIZE]

# # def preprocess_data(keypoints, muscle_group, exercise, fps):
# #     with buffer_lock:
# #         # Add new frame to buffer
# #         frame_buffer.append(np.array(keypoints))
        
# #         # Maintain buffer size according to detected FPS
# #         if len(frame_buffer) > BUFFER_SIZE * (fps / 30):
# #             frame_buffer.pop(0)
            
# #         # Resample if needed
# #         if abs(fps - 30) > 2:
# #             resampled = resample_frames(frame_buffer, fps)
# #         else:
# #             resampled = np.array(frame_buffer[-BUFFER_SIZE:])
        
# #         # Pad if insufficient frames
# #         if len(resampled) < BUFFER_SIZE:
# #             resampled = np.pad(resampled, ((0, BUFFER_SIZE-len(resampled)), (0,0), (0,0)), 
# #                            mode='edge')
        
# #         # Normalize
# #         mean = resampled.mean(axis=(0, 1))
# #         std = resampled.std(axis=(0, 1)) + 1e-8
# #         normalized = (resampled - mean) / std
        
# #         # Prepare other inputs
# #         muscle_encoded = muscle_encoder.transform([muscle_group])[0]
# #         exercise_encoded = exercise_encoder.transform([exercise])[0]
        
# #         return (
# #             normalized[np.newaxis, ...],  # Add batch dimension
# #             np.array([muscle_encoded]),
# #             np.array([exercise_encoded])
# #         )

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     data = request.json
# #     try:
# #         # Get input data
# #         keypoints = data['keypoints']
# #         muscle_group = data['muscle_group']
# #         exercise = data['exercise']
# #         fps = data.get('fps', 30)
        
# #         # Preprocess
# #         processed_data = preprocess_data(keypoints, muscle_group, exercise, fps)
        
# #         # Predict
# #         prediction = model.predict(processed_data)
# #         class_idx = np.argmax(prediction)
        
# #         # Get class label
# #         if class_idx == 0:
# #             return jsonify({'result': 'correct', 'confidence': float(prediction[0][class_idx])})
# #         else:
# #             return jsonify({
# #                 'result': 'incorrect',
# #                 'error': muscle_encoder.inverse_transform([class_idx])[0],
# #                 'confidence': float(prediction[0][class_idx])
# #             })
            
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 400

# # if __name__ == '__main__':
# #     app.run(host='0.0.0.0', port=5000, threaded=True)



# import numpy as np
# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# import joblib
# import os
# from scipy.interpolate import interp1d
# import threading
# import time

# import logging
# from logging.handlers import RotatingFileHandler

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         RotatingFileHandler('backend.log', maxBytes=1e6, backupCount=3),
#         logging.StreamHandler()
#     ]
# )

# app = Flask(__name__)

# # Load assets with absolute paths
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# model = load_model(os.path.join(BASE_DIR, 'assets', 'models', 'posture_model_final.keras'))
# muscle_encoder = joblib.load(os.path.join(BASE_DIR, 'assets', 'models', 'muscle_encoder.joblib'))
# exercise_encoder = joblib.load(os.path.join(BASE_DIR, 'assets', 'models', 'exercise_encoder.joblib'))

# # Session-based buffers for real-time processing
# sessions = {}
# session_lock = threading.Lock()

# def resample_frames(frames, original_fps, target_fps=30):
#     """Resample frames to target FPS using linear interpolation"""
#     original_time = np.linspace(0, 1, len(frames))
#     target_time = np.linspace(0, 1, int(len(frames) * target_fps / original_fps))
    
#     interpolator = interp1d(original_time, frames, axis=0, kind='linear')
#     resampled = interpolator(target_time)
#     return resampled[:BUFFER_SIZE]

# def process_session_data(session_id, keypoints, muscle_group, exercise, fps):
#     with session_lock:
#         if session_id not in sessions:
#             sessions[session_id] = {
#                 'buffer': [],
#                 'muscle_group': muscle_group,
#                 'exercise': exercise,
#                 'last_processed': time.time()
#             }
        
#         session = sessions[session_id]
#         session['buffer'].append(np.array(keypoints))
        
#         # Maintain buffer for 5 seconds of data
#         max_frames = int(5 * fps)
#         if len(session['buffer']) > max_frames:
#             session['buffer'] = session['buffer'][-max_frames:]
        
#         # Process every 5 frames (≈6fps processing)
#         if time.time() - session['last_processed'] > 0.16:  # ≈6Hz
#             # Resample to target 30fps
#             if abs(fps - 30) > 2:
#                 resampled = resample_frames(session['buffer'], fps)
#             else:
#                 resampled = np.array(session['buffer'])
            
#             # Pad if needed
#             if len(resampled) < 150:
#                 resampled = np.pad(resampled, ((0, 150-len(resampled)), (0,0), (0,0)), 
#                                 mode='edge')
            
#             # Normalize
#             mean = resampled.mean(axis=(0,1))
#             std = resampled.std(axis=(0,1)) + 1e-8
#             normalized = (resampled - mean) / std
            
#             # Prepare model inputs
#             muscle_encoded = muscle_encoder.transform([muscle_group])[0]
#             exercise_encoded = exercise_encoder.transform([exercise])[0]
            
#             session['last_processed'] = time.time()
#             return normalized[np.newaxis, ...], muscle_encoded, exercise_encoded
        
#         return None, None, None

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     data = request.json
# #     try:
# #         session_id = data.get('session_id', 'default')
# #         keypoints = data['keypoints']
# #         muscle_group = data['muscle_group']
# #         exercise = data['exercise']
# #         fps = data.get('fps', 30)
        
# #         # Process data
# #         processed_data, muscle_enc, exercise_enc = process_session_data(
# #             session_id, keypoints, muscle_group, exercise, fps
# #         )
        
# #         if processed_data is not None:

# #             # Added this code snippet
# #             if muscle_group not in muscle_encoder.classes_:
# #                 return jsonify({'error': 'Invalid muscle group'}), 400
# #             if exercise not in exercise_encoder.classes_:
# #                 return jsonify({'error': 'Invalid exercise'}), 400
            
# #             prediction = model.predict([processed_data, np.array([muscle_enc]), np.array([exercise_enc])])
# #             class_idx = np.argmax(prediction)
            
# #             # return jsonify({
# #             #     'result': 'correct' if class_idx == 0 else 'incorrect',
# #             #     'error': muscle_encoder.inverse_transform([class_idx])[0] if class_idx != 0 else '',
# #             #     'confidence': float(prediction[0][class_idx]),
# #             #     'timestamp': time.time()
# #             # })

# #             return jsonify({
# #                 'result': 'correct' if class_idx == 0 else 'incorrect',
# #                 'error': muscle_encoder.inverse_transform([class_idx])[0] if class_idx != 0 else '',
# #                 'confidence': float(prediction[0][class_idx]),
# #                 'timestamp': time.time()
# #             })
        
# #         # return jsonify({'status': 'buffering'})
# #         return jsonify({'status': 'buffering', 'buffered': len(session['buffer'])})
    
# #     # except Exception as e:
# #     #     return jsonify({'error': str(e)}), 400
# #     except Exception as e:
# #         return jsonify({
# #             'error': f'Processing error: {str(e)}',
# #             'type': 'SERVER_ERROR'
# #         }), 500


# @app.route('/predict', methods=['POST'])
# def predict():
#     start_time = time.time()
#     session_id = "default"
#     try:
#         data = request.json
#         session_id = data.get('session_id', 'default')
        
#         # Validate required fields
#         required_fields = ['keypoints', 'muscle_group', 'exercise']
#         if not all(field in data for field in required_fields):
#             logging.error(f"Missing fields in request: {data.keys()}")
#             return jsonify({'error': 'Missing required fields'}), 400

#         # Validate keypoints structure
#         keypoints = data['keypoints']
#         if (not isinstance(keypoints, list) or 
#             len(keypoints) == 0 or 
#             not all(len(point) == 3 for point in keypoints)):
#             logging.error(f"Invalid keypoints format: {type(keypoints)}")
#             return jsonify({'error': 'Invalid keypoints format'}), 400

#         # Validate muscle/exercise classes
#         muscle_group = data['muscle_group']
#         exercise = data['exercise']
#         if muscle_group not in muscle_encoder.classes_:
#             logging.error(f"Invalid muscle group: {muscle_group}")
#             return jsonify({'error': 'Invalid muscle group'}), 400
#         if exercise not in exercise_encoder.classes_:
#             logging.error(f"Invalid exercise: {exercise}")
#             return jsonify({'error': 'Invalid exercise'}), 400

#         # Process data
#         fps = data.get('fps', 30)
#         processed_data, muscle_enc, exercise_enc = process_session_data(
#             session_id, keypoints, muscle_group, exercise, fps
#         )
        
#         if processed_data is not None:
#             # Validate input shapes
#             if processed_data.shape[1:] != model.input_shape[1:]:
#                 logging.error(f"Input shape mismatch. Expected {model.input_shape[1:]}, got {processed_data.shape[1:]}")
#                 return jsonify({'error': 'Input shape mismatch'}), 400
            
#             # Make prediction
#             prediction = model.predict([processed_data, np.array([muscle_enc]), np.array([exercise_enc])])
#             class_idx = np.argmax(prediction)
            
#             return jsonify({
#                 'result': 'correct' if class_idx == 0 else 'incorrect',
#                 'error': muscle_encoder.inverse_transform([class_idx])[0] if class_idx != 0 else '',
#                 'confidence': float(prediction[0][class_idx]),
#                 'processing_time': time.time() - start_time
#             })
        
#         return jsonify({'status': 'buffering', 'buffered': len(sessions[session_id]['buffer'])})
        
#     except Exception as e:
#         logging.exception(f"Critical error in session {session_id}")
#         return jsonify({
#             'error': 'Internal server error',
#             'details': str(e),
#             'session': session_id
#         }), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, threaded=True)

# Add to top of backend.py
import sys
import logging
from logging import StreamHandler

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create handler that flushes immediately
class ImmediateFlushHandler(StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

handler = ImmediateFlushHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
# import logging
# from logging import StreamHandler

# # Configure logging before other imports
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[StreamHandler(stream=sys.stderr)]  # stderr flushes immediately
# )

import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
from tensorflow.keras.models import load_model
from typing import List, Optional
import json
import uvicorn

import threading
import traceback

# logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Configuration
MODEL_PATH = os.path.join('assets', 'models', 'posture_model_final.keras')
ENCODERS_PATH = os.path.join('assets', 'models')
BUFFER_SIZE = 150  # 5 seconds @ 30 FPS
MIN_FRAMES = 30    # Minimum frames for prediction

# Load resources
model = load_model(MODEL_PATH)
muscle_encoder = load(os.path.join(ENCODERS_PATH, 'muscle_encoder.joblib'))
exercise_encoder = load(os.path.join(ENCODERS_PATH, 'exercise_encoder.joblib'))

with open(os.path.join(ENCODERS_PATH, 'class_mapping.json'), 'r') as f:
    class_mapping = json.load(f)

# Reverse mapping for class labels
class_labels = {v: k for k, v in class_mapping.items()}

# Global buffer with metadata
frame_buffer = []
current_session = None

buffer_lock = threading.Lock()

class FrameData(BaseModel):
    landmarks: List[List[float]]
    muscle_group: str
    exercise: str
    timestamp: float

def preprocess_sequence(sequence: np.ndarray):
    """Preprocess sequence matching training pipeline"""
    # Normalize per sequence
    mean = np.mean(sequence, axis=(0, 1))
    std = np.std(sequence, axis=(0, 1)) + 1e-8
    return (sequence - mean) / std

# def predict_sliding_window(buffer: list):

#     try:
#         """Make predictions using sliding window approach"""
#         sequence = np.array([frame['landmarks'] for frame in buffer])
        
#         # Handle variable length sequences
#         if len(sequence) < BUFFER_SIZE:
#             padding = np.tile(sequence[-1], (BUFFER_SIZE - len(sequence), 1, 1))
#             sequence = np.concatenate([sequence, padding])
        
#         sequence = preprocess_sequence(sequence)
#         muscle_encoded = muscle_encoder.transform(buffer[0]['muscle_group'])[0]
#         exercise_encoded = exercise_encoder.transform(buffer[0]['exercise'])[0]
        
#         prediction = model.predict([
#             sequence[np.newaxis, ...],
#             np.array([muscle_encoded])[np.newaxis, ...],
#             np.array([exercise_encoded])[np.newaxis, ...]
#         ])
        
#         return class_labels[np.argmax(prediction[0])]
#     except ValueError as e:
#         logging.error(f"Invalid category: {e}")
#         return "unknown_error"


def predict_sliding_window(buffer: list):
    """Make predictions using sliding window approach"""
    try:
        # Validate input categories
        valid_muscles = list(muscle_encoder.classes_)
        valid_exercises = list(exercise_encoder.classes_)
        
        if buffer[0]['muscle_group'] not in valid_muscles:
            logging.error(f"Invalid muscle group: {buffer[0]['muscle_group']}")
            return "invalid_muscle_group"
            
        if buffer[0]['exercise'] not in valid_exercises:
            logging.error(f"Invalid exercise: {buffer[0]['exercise']}")
            return "invalid_exercise"

        # Process sequence
        sequence = np.array([frame['landmarks'] for frame in buffer])
        
        if len(sequence) < BUFFER_SIZE:
            padding = np.tile(sequence[-1], (BUFFER_SIZE - len(sequence), 1, 1))
            sequence = np.concatenate([sequence, padding])
        
        sequence = preprocess_sequence(sequence)
        
        # CORRECTED encoding with list wrapping
        muscle_encoded = muscle_encoder.transform([buffer[0]['muscle_group']])[0]
        exercise_encoded = exercise_encoder.transform([buffer[0]['exercise']])[0]
        
        prediction = model.predict([
            sequence[np.newaxis, ...],
            np.array([muscle_encoded])[np.newaxis, ...],
            np.array([exercise_encoded])[np.newaxis, ...]
        ])
        
        return class_labels[np.argmax(prediction[0])]
        
    except ValueError as e:
        logging.error(f"Encoding Error: {str(e)}")
        return "encoding_error"
    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        return "prediction_error"


@app.post("/process-frame")
async def process_frame(frames: List[FrameData]):
    global frame_buffer, current_session
    if not frames:
        raise HTTPException(status_code=400, detail="Empty request")

    # Validate frames (existing code)
    for frame in frames:
        if len(frame.landmarks) != 33 or any(len(lmk)!=3 for lmk in frame.landmarks):
            raise HTTPException(status_code=400, detail="Invalid landmarks format")

    with buffer_lock:  # Acquire lock for thread safety
        for frame in frames:
            current_id = f"{frame.muscle_group}_{frame.exercise}"
            if current_session != current_id:
                frame_buffer = []
                current_session = current_id

            frame_buffer.append({
                'landmarks': np.array(frame.landmarks),
                'muscle_group': frame.muscle_group,
                'exercise': frame.exercise,
                'timestamp': frame.timestamp
            })

        # Maintain buffer size
        if len(frame_buffer) > BUFFER_SIZE * 2:
            frame_buffer = frame_buffer[-BUFFER_SIZE:]

        # Make prediction
        if len(frame_buffer) >= MIN_FRAMES:
            try:
                # Use a copy to avoid changes during prediction
                buffer_copy = frame_buffer[-BUFFER_SIZE:].copy()
                prediction = predict_sliding_window(buffer_copy)
                logger.debug(f"MODEL PREDICTION: {prediction}")
                # logging.debug(f"MODEL PREDICTION: {prediction}")  # Correct logging
                return {"prediction": prediction, "buffer": len(frame_buffer)}
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))
    
    return {"status": "buffering", "buffer": len(frame_buffer)}


# @app.post("/process-frame")
# async def process_frame(frames: List[FrameData]):
#     global frame_buffer, current_session
#     if not frames:
#         raise HTTPException(status_code=400, detail="Empty request")

#     # Validate all frames
#     for frame in frames:
#         if len(frame.landmarks) != 33 or any(len(lmk)!=3 for lmk in frame.landmarks):
#             raise HTTPException(status_code=400, detail="Invalid landmarks format")

#     # Process batch
#     for frame in frames:
#         current_id = f"{frame.muscle_group}_{frame.exercise}"
#         if current_session != current_id:
#             frame_buffer = []
#             current_session = current_id

#         frame_buffer.append({
#             'landmarks': np.array(frame.landmarks),
#             'muscle_group': frame.muscle_group,
#             'exercise': frame.exercise,
#             'timestamp': frame.timestamp
#         })
    
#     # Maintain buffer size
#     if len(frame_buffer) > BUFFER_SIZE * 2:
#         frame_buffer = frame_buffer[-BUFFER_SIZE:]
    
#     # Make prediction if sufficient data
#     if len(frame_buffer) >= MIN_FRAMES:
#         try:
#             prediction = predict_sliding_window(frame_buffer[-BUFFER_SIZE:])
#             logging.debug("MODEL PREDICTION: ", prediction)

#             return {"prediction": prediction, "buffer": len(frame_buffer)}
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))
    
#     return {"status": "buffering", "buffer": len(frame_buffer)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)