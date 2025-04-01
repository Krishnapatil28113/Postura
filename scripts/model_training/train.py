import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, LSTM, Dense, TimeDistributed, Concatenate, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from scripts.utils.normalization import load_normalized_data

def build_hybrid_model(num_frames, num_joints, num_coords, num_exercises, num_classes):
    # Keypoints input branch
    kp_input = Input(shape=(num_frames, num_joints, num_coords), name='keypoints_input')
    
    # CNN for spatial features
    cnn = TimeDistributed(Conv1D(64, 3, activation='relu', padding='same'))
    cnn = TimeDistributed(MaxPooling1D(2))
    cnn = TimeDistributed(Conv1D(128, 3, activation='relu', padding='same'))
    cnn = TimeDistributed(MaxPooling1D(2))
    cnn = TimeDistributed(Flatten())
    
    x = cnn(kp_input)
    
    # LSTM for temporal features
    x = LSTM(256, return_sequences=True, dropout=0.3)(x)
    x = LSTM(128, dropout=0.2)(x)
    
    # Exercise input branch
    exercise_input = Input(shape=(num_exercises,), name='exercise_input')
    merged = Concatenate()([x, exercise_input])
    
    # Classification head
    x = Dense(256, activation='relu')(merged)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=[kp_input, exercise_input], outputs=output)
    return model

def train_model():
    # Configuration
    MAX_FRAMES = 150  # Based on dataset analysis
    BATCH_SIZE = 32
    EPOCHS = 100
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Load and preprocess data
    labels_path = os.path.join('assets', 'datasets', 'raw', 'labels.csv')
    dataset_root = os.path.join('assets', 'datasets', 'raw')
    
    # Load dataset
    df = pd.read_csv(labels_path)
    df['filepath'] = df['filepath'].apply(lambda x: os.path.join(
        dataset_root,
        x.replace('\\', '/')  # Convert Windows paths to Unix-style
    ))
    
    # Create label mapping
    unique_errors = df[df['is_correct'] == 0]['errors'].unique()
    class_mapping = {'correct': 0}
    class_mapping.update({error: i+1 for i, error in enumerate(unique_errors)})
    
    # Prepare labels
    y = np.array([class_mapping['correct'] if row['is_correct'] else class_mapping[row['errors']]
                  for _, row in df.iterrows()])
    
    # Prepare exercise encoding
    exercise_encoder = LabelEncoder()
    exercise_encoded = exercise_encoder.fit_transform(df['exercise'])
    exercise_onehot = to_categorical(exercise_encoded)
    
    # Load and preprocess keypoints
    X_kp = []
    for path in df['filepath']:
        path = path.replace('\\', '/').strip()
        try:
            kp = np.load(path)
        except FileNotFoundError:
            print(f"File not found: {path}")
            continue
            
        # Pad/truncate to MAX_FRAMES
        if kp.shape[0] < MAX_FRAMES:
            pad = np.zeros((MAX_FRAMES - kp.shape[0], *kp.shape[1:]))
            kp = np.vstack([kp, pad])
        else:
            kp = kp[:MAX_FRAMES]
        X_kp.append(kp)
    
    X_kp = np.array(X_kp)
    
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        np.arange(len(y)), y, test_size=TEST_SPLIT + VAL_SPLIT, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SPLIT/(TEST_SPLIT + VAL_SPLIT), stratify=y_temp)
    
    # Build model
    model = build_hybrid_model(
        num_frames=MAX_FRAMES,
        num_joints=33,
        num_coords=3,
        num_exercises=len(exercise_encoder.classes_),
        num_classes=len(class_mapping)
    )
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'assets/models/posture_model.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        x=[X_kp[X_train], exercise_onehot[X_train]],
        y=y_train,
        validation_data=([X_kp[X_val], exercise_onehot[X_val]], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stop],
        class_weight=compute_class_weight(y_train)  # Implement this based on your data distribution
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(
        [X_kp[X_test], exercise_onehot[X_test]], y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    
    # Save final model
    model.save('assets/models/posture_model_final.h5')

def compute_class_weight(y):
    classes = np.unique(y)
    class_weights = {}
    total = len(y)
    for cls in classes:
        count = np.sum(y == cls)
        class_weights[cls] = (1 / count) * (total / len(classes)) * 0.5
    return class_weights

if __name__ == "__main__":
    train_model()







