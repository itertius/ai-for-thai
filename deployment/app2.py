import streamlit as st
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import cv2
import tempfile
import os

# --- Setup MediaPipe Pose ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# --- UI Header ---
st.title("🏋️ Powerlifting Pose Classification from Video")

# --- Select Exercise Type ---
exercise_type = st.selectbox("Select Exercise Type", ["Benchpress", "Deadlift", "Squat"])

# --- Load Corresponding Model ---
model_path = f"models/{exercise_type.lower()}/{exercise_type.lower()}.pkl"
if not os.path.exists(model_path):
    st.error(f"Model for {exercise_type} not found at {model_path}")
    st.stop()

with open(model_path, "rb") as f:
    model = pickle.load(f)

# --- Video Source Options ---
st.subheader("🎥 Choose Video Source")
use_sample = st.checkbox("Use sample video")
enable_camera = st.checkbox("Enable camera")

video_path = None

if use_sample:
    sample_paths = {
        "Squat Sample": "sample/squat_sample.mp4",
        "Deadlift Sample": "sample/deadlift_sample.mp4",
        "Benchpress Sample": "sample/benchpress_sample.mp4"
    }
    selected_sample = st.selectbox("Select a sample video", list(sample_paths.keys()))
    video_path = sample_paths[selected_sample]

    if not os.path.exists(video_path):
        st.error(f"Sample video not found at `{video_path}`")
        st.stop()

elif enable_camera:
    video_path = 0  # Webcam (OpenCV default)

else:
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
    else:
        st.warning("Please upload a video or select a sample/camera option.")
        st.stop()

# --- Extract Pose Landmarks ---
def extract_pose_landmarks(results):
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    row = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z, lm.visibility)]
    return row if len(row) == 132 else None

# --- Process Video ---
cap = cv2.VideoCapture(video_path)
frame_count = 0
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not enable_camera else 100  # arbitrary for webcam
predicted_classes = []

stframe = st.empty()
progress_bar = st.progress(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Get prediction
    landmarks = extract_pose_landmarks(results)
    if landmarks:
        input_np = np.array(landmarks).reshape(1, -1)
        pred_class = model.predict(input_np)[0]
        predicted_classes.append(pred_class)

        # Draw pose landmarks and prediction text
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, f"Predicted: {pred_class}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frame
    stframe.image(frame, channels="BGR")

    # Update progress bar
    frame_count += 1
    if not enable_camera:
        progress = int((frame_count / frame_total) * 100)
        progress_bar.progress(min(progress, 100))

    if enable_camera and frame_count >= 100:
        break

cap.release()

# --- Display Final Results ---
if predicted_classes:
    final_class = max(set(predicted_classes), key=predicted_classes.count)
    st.success(f"✅ Final Prediction for {exercise_type}: **{final_class}**")

    st.write("🧮 Frame-by-frame predictions:")
    st.write(predicted_classes)

    # Download as CSV
    df = pd.DataFrame({
        "Frame": range(len(predicted_classes)),
        "Prediction": predicted_classes
    })
    st.download_button("📥 Download Predictions as CSV", df.to_csv(index=False),
                       file_name="predictions.csv", mime="text/csv")
