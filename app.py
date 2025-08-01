import streamlit as st
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import cv2
import tempfile
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# --- UI Header ---
st.title("🏋️ Powerlifting Pose Classification from Video")

# --- Select Exercise ---
exercise_type = st.selectbox("Select Exercise Type", ["Benchpress", "Deadlift", "Squat"])

# --- Load Corresponding Model ---
if exercise_type == "Benchpress":
    with open("models/benchpress/benchpress.pkl", "rb") as f:
        model = pickle.load(f)
elif exercise_type == "Deadlift":
    with open("models/deadlift/deadlift.pkl", "rb") as f:
        model = pickle.load(f)
else:
    with open("models/squat/squat.pkl", "rb") as f:
        model = pickle.load(f)

# --- Choose Video Source ---
use_sample = st.checkbox("Use sample video instead of uploading")

if use_sample:
    sample_path = "sample/squat_sample.mp4"
    if not os.path.exists(sample_path):
        st.error("Sample video not found! Please put a video at `sample/squat_sample.mp4`")
        st.stop()
    video_path = sample_path
else:
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name
    else:
        st.warning("Please upload a video or check 'Use sample video'")
        st.stop()

# --- Pose Extraction Function ---
def extract_pose_landmarks(results):
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    row = []
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    return row if len(row) == 132 else None

# --- Start Video Processing ---
cap = cv2.VideoCapture(video_path)
frame_count = 0
predicted_classes = []
stframe = st.empty()
progress_bar = st.progress(0)
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Prediction
    landmarks = extract_pose_landmarks(results)
    if landmarks:
        input_np = np.array(landmarks).reshape(1, -1)
        pred_class = model.predict(input_np)[0]
        predicted_classes.append(pred_class)

        # Draw pose & prediction
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, f"Predicted: {pred_class}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show on Streamlit
    stframe.image(frame, channels="BGR")

    # Progress
    frame_count += 1
    progress = min(int((frame_count / frame_total) * 100), 100)
    progress_bar.progress(progress)

cap.release()

# --- Summary Result ---
if predicted_classes:
    final_class = max(set(predicted_classes), key=predicted_classes.count)
    st.success(f"✅ Final Prediction for {exercise_type}: **{final_class}**")
    st.write("🧮 All Frame Predictions:")
    st.write(predicted_classes)

    # Download as CSV
    df_results = pd.DataFrame({
        "Frame": list(range(len(predicted_classes))),
        "Prediction": predicted_classes
    })
    st.download_button("📥 Download Predictions as CSV",
                       df_results.to_csv(index=False),
                       "predictions.csv",
                       "text/csv")