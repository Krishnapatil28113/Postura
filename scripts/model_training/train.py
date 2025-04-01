# import os
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, TimeDistributed, Concatenate, Dropout
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.keras.optimizers import Adam
# from scripts.utils.normalization import load_normalized_data

# def build_hybrid_model(num_frames, num_joints, num_coords, num_exercises, num_classes):
#     # Keypoints input branch
#     kp_input = Input(shape=(num_frames, num_joints, num_coords), name='keypoints_input')
    
#     # CNN for spatial features
#     cnn = TimeDistributed(Conv1D(64, 3, activation='relu', padding='same'))
#     cnn = TimeDistributed(MaxPooling1D(2))
#     cnn = TimeDistributed(Conv1D(128, 3, activation='relu', padding='same'))
#     cnn = TimeDistributed(MaxPooling1D(2))
#     cnn = TimeDistributed(Flatten())
    
#     x = cnn(kp_input)
    
#     # LSTM for temporal features
#     x = LSTM(256, return_sequences=True, dropout=0.3)(x)
#     x = LSTM(128, dropout=0.2)(x)
    
#     # Exercise input branch
#     exercise_input = Input(shape=(num_exercises,), name='exercise_input')
#     merged = Concatenate()([x, exercise_input])
    
#     # Classification head
#     x = Dense(256, activation='relu')(merged)
#     x = Dropout(0.4)(x)
#     x = Dense(128, activation='relu')(x)
#     output = Dense(num_classes, activation='softmax')(x)
    
#     model = Model(inputs=[kp_input, exercise_input], outputs=output)
#     return model

# def train_model():
#     # Configuration
#     MAX_FRAMES = 150  # Based on dataset analysis
#     BATCH_SIZE = 32
#     EPOCHS = 100
#     VAL_SPLIT = 0.15
#     TEST_SPLIT = 0.15
    
#     # Load and preprocess data
#     labels_path = os.path.join('assets', 'datasets', 'raw', 'labels.csv')
#     dataset_root = os.path.join('assets', 'datasets', 'raw')
    
#     # Load dataset
#     df = pd.read_csv(labels_path)
#     df['filepath'] = df['filepath'].apply(lambda x: os.path.join(
#         dataset_root,
#         x.replace('\\', '/')  # Convert Windows paths to Unix-style
#     ))
    
#     # Create label mapping
#     unique_errors = df[df['is_correct'] == 0]['errors'].unique()
#     class_mapping = {'correct': 0}
#     class_mapping.update({error: i+1 for i, error in enumerate(unique_errors)})
    
#     # Prepare labels
#     y = np.array([class_mapping['correct'] if row['is_correct'] else class_mapping[row['errors']]
#                   for _, row in df.iterrows()])
    
#     # Prepare exercise encoding
#     exercise_encoder = LabelEncoder()
#     exercise_encoded = exercise_encoder.fit_transform(df['exercise'])
#     exercise_onehot = to_categorical(exercise_encoded)
    
#     # Load and preprocess keypoints
#     X_kp = []
#     for path in df['filepath']:
#         path = path.replace('\\', '/').strip()
#         try:
#             kp = np.load(path)
#         except FileNotFoundError:
#             print(f"File not found: {path}")
#             continue
            
#         # Pad/truncate to MAX_FRAMES
#         if kp.shape[0] < MAX_FRAMES:
#             pad = np.zeros((MAX_FRAMES - kp.shape[0], *kp.shape[1:]))
#             kp = np.vstack([kp, pad])
#         else:
#             kp = kp[:MAX_FRAMES]
#         X_kp.append(kp)
    
#     X_kp = np.array(X_kp)
    
#     # Split dataset
#     X_train, X_temp, y_train, y_temp = train_test_split(
#         np.arange(len(y)), y, test_size=TEST_SPLIT + VAL_SPLIT, stratify=y)
#     X_val, X_test, y_val, y_test = train_test_split(
#         X_temp, y_temp, test_size=TEST_SPLIT/(TEST_SPLIT + VAL_SPLIT), stratify=y_temp)
    
#     # Build model
#     model = build_hybrid_model(
#         num_frames=MAX_FRAMES,
#         num_joints=33,
#         num_coords=3,
#         num_exercises=len(exercise_encoder.classes_),
#         num_classes=len(class_mapping)
#     )
    
#     model.compile(
#         optimizer=Adam(learning_rate=0.001),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     # Callbacks
#     checkpoint = ModelCheckpoint(
#         'assets/models/posture_model.h5',
#         save_best_only=True,
#         monitor='val_accuracy',
#         mode='max'
#     )
#     early_stop = EarlyStopping(
#         monitor='val_loss',
#         patience=15,
#         restore_best_weights=True
#     )
    
#     # Train model
#     history = model.fit(
#         x=[X_kp[X_train], exercise_onehot[X_train]],
#         y=y_train,
#         validation_data=([X_kp[X_val], exercise_onehot[X_val]], y_val),
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         callbacks=[checkpoint, early_stop],
#         class_weight=compute_class_weight(y_train)  # Implement this based on your data distribution
#     )
    
#     # Evaluate
#     test_loss, test_acc = model.evaluate(
#         [X_kp[X_test], exercise_onehot[X_test]], y_test)
#     print(f"\nTest Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    
#     # Save final model
#     model.save('assets/models/posture_model_final.h5')

# def compute_class_weight(y):
#     classes = np.unique(y)
#     class_weights = {}
#     total = len(y)
#     for cls in classes:
#         count = np.sum(y == cls)
#         class_weights[cls] = (1 / count) * (total / len(classes)) * 0.5
#     return class_weights

# if __name__ == "__main__":
#     train_model()


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, LSTM, Dense,
    TimeDistributed, Concatenate, Dropout, Embedding,
    Bidirectional, BatchNormalization, Attention, LayerNormalization,
    SpatialDropout1D, GaussianNoise
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from scipy.spatial.transform import Rotation
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Enhanced Configuration
MAX_FRAMES = 150  # 5 seconds at 30 FPS
BATCH_SIZE = 16  # Increased batch size
EPOCHS = 100
LEARNING_RATE = 0.0005
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
DROPOUT_RATE = 0.5
L2_REG = 0.001

def load_and_sample_keypoints(path, max_frames=150):
    """Enhanced data loading with more augmentation"""
    try:
        kp = np.load(path.strip())

        # Enhanced Data Augmentation
        # Random rotation
        angle = np.random.uniform(-15, 15)  # Increased rotation range
        rotation_matrix = Rotation.from_euler('zxy', [
            np.random.uniform(-5, 5),
            np.random.uniform(-5, 5),
            angle
        ], degrees=True).as_matrix()
        kp = np.dot(kp, rotation_matrix)

        # Random translation
        translation = np.random.uniform(-0.1, 0.1, size=(3,))  # Increased translation
        kp += translation

        # Random scaling
        scale = np.random.uniform(0.9, 1.1)
        kp *= scale

        # Add Gaussian noise
        kp += np.random.normal(0, 0.01, size=kp.shape)

        # Temporal sampling with random warping
        if len(kp) > max_frames:
            # Random temporal warping
            start = np.random.randint(0, len(kp)-max_frames)
            step = np.random.choice([1, 2])
            kp = kp[start:start+max_frames:step]
            if len(kp) < max_frames:
                kp = np.concatenate([kp, np.tile(kp[-1], (max_frames-len(kp),1,1))])
        else:
            # Pad with interpolated frames
            diff = max_frames - len(kp)
            for _ in range(diff):
                insert_idx = np.random.randint(0, len(kp))
                new_frame = (kp[insert_idx] + kp[insert_idx-1])/2
                kp = np.insert(kp, insert_idx, new_frame, axis=0)
            kp = kp[:max_frames]


        kp_mean = np.mean(kp, axis=(0, 1))
        kp_std = np.std(kp, axis=(0, 1)) + 1e-8
        kp = (kp - kp_mean) / kp_std
        return kp
    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        return None


def build_hybrid_model(num_frames, num_joints, num_coords, num_muscle_groups, num_exercises, num_classes):
    # Keypoints input branch
    kp_input = Input(shape=(num_frames, num_joints, num_coords), name='keypoints_input')

    # Simplified CNN architecture
    x = TimeDistributed(Conv1D(128, 5, activation='relu', padding='same'))(kp_input)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling1D(2))(x)

    x = TimeDistributed(Conv1D(256, 3, activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling1D(2))(x)

    x = TimeDistributed(Flatten())(x)

    # Simplified LSTM with reduced units
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(128))(x)

    # Embedding layers with reduced dimensions
    muscle_input = Input(shape=(1,), name='muscle_input')
    exercise_input = Input(shape=(1,), name='exercise_input')

    muscle_embed = Embedding(num_muscle_groups, 32)(muscle_input)
    exercise_embed = Embedding(num_exercises, 32)(exercise_input)

    # Concatenate features
    merged = Concatenate()([
        x,
        Flatten()(muscle_embed),
        Flatten()(exercise_embed)
    ])

    # Simplified classifier
    x = Dense(256, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    return Model(
        inputs=[kp_input, muscle_input, exercise_input],
        outputs=output
    )

def compute_class_weight(y):
    """Enhanced class weighting using inverse square root frequency balancing"""
    classes, counts = np.unique(y, return_counts=True)
    weights = np.sqrt(np.median(counts)/counts)
    return {cls: weight for cls, weight in zip(classes, weights)}

def train_model():
    # Path configuration
    base_dir = '/content/drive/MyDrive/assets/'
    labels_path = os.path.join(base_dir, 'datasets/raw/labels.csv')
    dataset_root = os.path.join(base_dir, 'datasets/raw/')

    # Load and prepare data
    df = pd.read_csv(labels_path)
    df['filepath'] = df['filepath'].apply(
        lambda x: os.path.join(dataset_root, x.replace('\\', '/').lstrip('/'))
    )

    # Encode categorical features
    muscle_encoder = LabelEncoder()
    exercise_encoder = LabelEncoder()
    df['muscle_encoded'] = muscle_encoder.fit_transform(df['muscle_group'])
    df['exercise_encoded'] = exercise_encoder.fit_transform(df['exercise'])

    # Prepare labels
    unique_errors = df[df['is_correct'] == 0]['errors'].unique()
    class_mapping = {'correct': 0}
    class_mapping.update({error: i+1 for i, error in enumerate(unique_errors)})
    y = np.array([class_mapping['correct'] if row['is_correct'] else class_mapping[row['errors']]
                  for _, row in df.iterrows()])

    # Load and preprocess keypoints
    X_kp = []
    valid_indices = []
    for idx, path in enumerate(df['filepath']):
        kp = load_and_sample_keypoints(path, MAX_FRAMES)
        if kp is not None and len(kp) == MAX_FRAMES:
            X_kp.append(kp)
            valid_indices.append(idx)

    # Filter valid samples
    df = df.iloc[valid_indices].reset_index(drop=True)
    y = y[valid_indices]
    X_kp = np.array(X_kp)

    # Per-sample normalization
    for i in range(len(X_kp)):
        mean = X_kp[i].mean(axis=(0,1))
        std = X_kp[i].std(axis=(0,1)) + 1e-8
        X_kp[i] = (X_kp[i] - mean) / std

    # Stratified split with multiple labels
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=TEST_SPLIT,
        stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=VAL_SPLIT/(1-TEST_SPLIT),
        stratify=y[train_idx]
    )

    # Build model
    model = build_hybrid_model(
        num_frames=MAX_FRAMES,
        num_joints=33,
        num_coords=3,
        num_muscle_groups=len(muscle_encoder.classes_),
        num_exercises=len(exercise_encoder.classes_),
        num_classes=len(class_mapping)
    )

    # Optimizer with gradient clipping
    optimizer = Nadam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0
    )
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=CategoricalFocalCrossentropy(gamma=2.0, alpha=0.25),
        metrics=['accuracy'],
        weighted_metrics=['accuracy']
    )

    # Enhanced callbacks
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            os.path.join(model_dir, 'posture_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            save_weights_only=False
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            min_delta=0.001,
            mode='max',
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        TensorBoard(
            log_dir=os.path.join(model_dir, 'logs'),
            histogram_freq=1,
            profile_batch=0
        )
    ]

    # Class weights
    class_weights = compute_class_weight(y[train_idx])

    # Train model with larger batches
    history = model.fit(
        x=[
            X_kp[train_idx],
            df.iloc[train_idx]['muscle_encoded'].values,
            df.iloc[train_idx]['exercise_encoded'].values
        ],
        y=to_categorical(y[train_idx]),
        validation_data=([
            X_kp[val_idx],
            df.iloc[val_idx]['muscle_encoded'].values,
            df.iloc[val_idx]['exercise_encoded'].values
        ], to_categorical(y[val_idx])),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        shuffle=True,
        verbose=2
    )

    # Evaluate with test set
    results = model.evaluate([
        X_kp[test_idx],
        df.iloc[test_idx]['muscle_encoded'].values,
        df.iloc[test_idx]['exercise_encoded'].values
    ], to_categorical(y[test_idx]))
    print(f"\nTest Results - Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}, Weighted Accuracy: {results[2]:.4f}")

    # Save final model
    model.save(os.path.join(model_dir, 'posture_model_final.keras'))

if __name__ == "__main__":
    train_model()