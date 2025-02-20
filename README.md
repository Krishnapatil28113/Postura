# Postura
A mobile application that analyzes gym exercises, yoga, and everyday postures from user-uploaded videos, providing real-time feedback for posture correction and improvement.



# Architecture Diagram

```
+-----------------------------+
| Presentation/UI Layer       |
| Flutter (Dart)              |
+-----------------------------+
            ↓
+-----------------------------+
| Input Capture & Pre-process |
| OpenCV                      |
+-----------------------------+
            ↓
+-----------------------------+
| Pose Estimation Layer       |
| MediaPipe                   |
+-----------------------------+
            ↓
+-----------------------------+
| Feature Extraction & Norm   |
| Dart Code                   |
+-----------------------------+
            ↓
+-----------------------------+
| Posture Analysis/ML Infer   |
| TensorFlow                  |
+-----------------------------+
            ↓
+-----------------------------+
| Feedback Generation/Visual  |
| Flutter                     |
+-----------------------------+
            ↓
+-----------------------------+
| Data Management & Cloud     |
| Firebase Suite:             |
|  • Firestore (NoSQL)        |
|  • Cloud Storage            |
|  • Authentication           |
|  • Cloud Functions          |
+-----------------------------+

```

# Overview of the Diagram

## **Presentation / UI Layer**
- This is the **front-end** of your mobile application.
- Users can:
  - Select the type of analysis (**Gym, Yoga, or Overall Posture**).
  - Interact with the app (e.g., upload or record a video).
- Built using **Flutter (Dart)** for a seamless and responsive user experience.

---

## **Input Capture & Pre-Processing Layer**
- Responsible for:
  - Capturing video input (uploaded or recorded).
  - Extracting frames from the video.
  - Performing pre-processing tasks like **resizing** and **normalization**.
- Uses **OpenCV** for efficient frame extraction and pre-processing.

---

## **Pose Estimation Layer**
- Detects the user's skeleton keypoints (joints) from each pre-processed frame.
- Uses models like **MediaPipe Pose** or **MoveNet** for accurate pose estimation.
- Outputs raw keypoints for further processing.

---

## **Feature Extraction & Normalization Layer**
- Processes raw keypoints to calculate higher-level features:
  - **Joint angles**.
  - **Distances between joints**.
- Normalizes features for consistent analysis across different users and scenarios.
- Implemented using **Dart Code** for seamless integration with the Flutter UI.

---

## **Posture Analysis / ML Inference Layer**
- Contains **exercise-specific machine learning models** or **rule-based models**.
- Evaluates extracted features to determine if the posture is correct.
- Classifies or scores the posture based on the selected exercise or pose.
- Built using **TensorFlow** for robust ML inference.

---

## **Feedback Generation & Visualization Layer**
- Converts analysis results into actionable feedback.
- Provides:
  - **Visual overlays** (e.g., highlighting joints, drawing angles).
  - **Textual or audio feedback** to guide the user.
- Feedback can be delivered in **real-time** or after analysis.
- Implemented using **Flutter** for a rich and interactive user experience.

---

## **Data Management & Cloud Services Layer**
- Handles:
  - **Storage** of user data and media.
  - **Analytics** for continuous improvement.
  - **Remote model updates** for scalability.
  - **Cloud-based processing** to complement on-device processing.
- Uses **Firebase Suite**:
  - **Firestore (NoSQL)**: For real-time data syncing, offline support, and scalability.
  - **Cloud Storage**: For securely storing user-uploaded videos and media.
  - **Authentication**: For secure user sign-in and identity management.
  - **Cloud Functions**: For running backend logic and processing data asynchronously.
- This layer is **optional/hybrid** and enhances scalability and remote capabilities.

---

# Folder Structure

```
gym_analysis_app/
│
├── lib/                          # Flutter Presentation/UI Layer
│   ├── src/
│   │   ├── app/                  # Core app setup
│   │   │   ├── app.dart
│   │   │   └── router.dart       # Navigation/routing
│   │   │
│   │   ├── features/
│   │   │   ├── authentication/   # Auth screens
│   │   │   ├── exercise_selection/ # Muscle/exercise selection UI
│   │   │   ├── camera_feed/      # Live camera interface
│   │   │   ├── video_analysis/   # Results visualization
│   │   │   └── progress_tracking/ # User progress dashboards
│   │   │
│   │   ├── widgets/              # Reusable components
│   │   │   ├── pose_skeleton_viewer.dart
│   │   │   └── exercise_card.dart
│   │   │
│   │   ├── services/
│   │   │   ├── firebase/         # Firebase integration
│   │   │   │   ├── firestore.dart
│   │   │   │   └── storage.dart
│   │   │   │
│   │   │   ├── ml/               # ML Inference Layer
│   │   │   │   ├── pose_processor.dart
│   │   │   │   └── model_handler.dart
│   │   │   │
│   │   │   └── auth/             # Authentication service
│   │   │
│   │   └── utils/                # Helpers/utilities
│   │       ├── pose_utils.dart   # Feature normalization
│   │       └── video_utils.dart  # Video processing
│   │
│   └── main.dart                 # App entry point
│
├── native/                       # Platform-specific code
│   ├── android/
│   │   ├── opencv/               # OpenCV preprocessing
│   │   └── mediapipe/            # Android MediaPipe setup
│   │
│   └── ios/
│       ├── opencv/               # iOS OpenCV config
│       └── mediapipe/            # iOS MediaPipe setup
│
├── assets/
│   ├── models/                   # TensorFlow models
│   │   ├── posture_lstm.tflite
│   │   └── exercise_classifier.tflite
│   │
│   ├── instructions/             # Exercise guides
│   ├── icons/                    # App icons
│   └── animations/               # Lottie animations
│
├── cloud_functions/              # Firebase Cloud Functions
│   ├── src/
│   │   ├── model_updater/        # Model switching logic
│   │   └── video_analysis/       # Heavy processing
│   └── index.ts                  # Main entry
│
├── test/                         # Unit/widget tests
│   └── ...
│
├── integration_test/             # Integration tests
│   └── ...
│
├── scripts/                      # Utility scripts
│   ├── model_converter/          # TF → TFLite conversion
│   └── data_preprocessing/       # Training data prep
│
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md
│   └── DATA_FLOW.md
│
└── pubspec.yaml                  # Flutter dependencies

```