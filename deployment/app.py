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
    with open("../models/benchpress/benchpress.pkl", "rb") as f:
        model = pickle.load(f)
elif exercise_type == "Deadlift":
    with open("../models/deadlift/deadlift.pkl", "rb") as f:
        model = pickle.load(f)
else:
    with open("../models/squat/squat.pkl", "rb") as f:
        model = pickle.load(f)

# --- Choose Video Source ---
use_sample = st.checkbox("Use sample video instead of uploading")

if use_sample:
    sample_options = {
        "Squat Sample": "../sample/squat_sample.mp4",
        "Deadlift Sample": "../sample/deadlift_sample.mp4",
        "Benchpress Sample": "../sample/benchpress_sample.mp4"
    }

    selected_sample = st.selectbox("Select a sample video", list(sample_options.keys()))
    sample_path = sample_options[selected_sample]

    if not os.path.exists(sample_path):
        st.error(f"Sample video not found! Please put a video at `{sample_path}`")
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

# --- Calculate Angles ---
def calculateAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- Start Video Processing ---
cap = cv2.VideoCapture(video_path)
frame_count = 0
predicted_classes = []
angle_data = []
stframe = st.empty()
progress_bar = st.progress(0)
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Prediction and Drawing
    landmarks = extract_pose_landmarks(results)
    if landmarks:
        input_np = np.array(landmarks).reshape(1, -1)
        pred_class = model.predict(input_np)[0]
        predicted_classes.append(pred_class)

        # Draw pose
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark

        def get_point(index):
            return [lm[index].x * frame.shape[1], lm[index].y * frame.shape[0]]

        try:
            left_elbow = calculateAngle(get_point(11), get_point(13), get_point(15))
            right_elbow = calculateAngle(get_point(12), get_point(14), get_point(16))
            left_knee = calculateAngle(get_point(23), get_point(25), get_point(27))
            right_knee = calculateAngle(get_point(24), get_point(26), get_point(28))
            left_shoulder = calculateAngle(get_point(13), get_point(11), get_point(23))
            right_shoulder = calculateAngle(get_point(14), get_point(12), get_point(24))

            # Display angles
            cv2.putText(frame, f"L_Elbow: {int(left_elbow)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            cv2.putText(frame, f"R_Elbow: {int(right_elbow)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            cv2.putText(frame, f"L_Knee: {int(left_knee)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"R_Knee: {int(right_knee)}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"L_Shoulder: {int(left_shoulder)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"R_Shoulder: {int(right_shoulder)}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            angle_data.append({
                "frame": frame_count,
                "left_elbow": left_elbow,
                "right_elbow": right_elbow,
                "left_knee": left_knee,
                "right_knee": right_knee,
                "left_shoulder": left_shoulder,
                "right_shoulder": right_shoulder
            })

        except IndexError:
            pass

        # Display prediction
        cv2.putText(frame, f"Predicted: {pred_class}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    stframe.image(frame, channels="BGR")

    # Progress bar
    frame_count += 1
    progress = min(int((frame_count / frame_total) * 100), 100)
    progress_bar.progress(progress)

cap.release()

# --- Summary Results ---
if predicted_classes:
    final_class = max(set(predicted_classes), key=predicted_classes.count)
    st.success(f"✅ Final Prediction for {exercise_type}: **{final_class}**")

    st.write("🧮 All Frame Predictions:")
    st.write(predicted_classes)

    df_results = pd.DataFrame({
        "Frame": list(range(len(predicted_classes))),
        "Prediction": predicted_classes
    })

    st.download_button("📥 Download Predictions as CSV",
                       df_results.to_csv(index=False),
                       "predictions.csv",
                       "text/csv")

if angle_data:
    df_angles = pd.DataFrame(angle_data)
    st.write("📐 Joint Angles per Frame:")
    st.dataframe(df_angles)
    st.download_button("📥 Download Joint Angles as CSV",
                       df_angles.to_csv(index=False),
                       "angles.csv",
                       "text/csv")
