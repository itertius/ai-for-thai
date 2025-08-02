import streamlit as st
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import tempfile
import os
import cv2
from transformers import pipeline
import requests

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# --- UI ---
st.title("🏋️ Powerlifting Pose Classification from Video with AI Feedback & TTS")

exercise_type = st.selectbox("Select Exercise Type", ["Benchpress", "Deadlift", "Squat"])

model_paths = {
    "Benchpress": "models/benchpress/benchpress.pkl",
    "Deadlift": "models/deadlift/deadlift.pkl",
    "Squat": "models/squat/squat.pkl"
}

with open(model_paths[exercise_type], "rb") as f:
    model = pickle.load(f)

use_sample = st.checkbox("Use sample video instead of upload")
if use_sample:
    sample_videos = {
        "Squat Sample": "sample/squat_sample.mp4",
        "Deadlift Sample": "sample/deadlift_sample.mp4",
        "Benchpress Sample": "sample/benchpress_sample.mp4"
    }
    sample_choice = st.selectbox("Choose sample video", list(sample_videos.keys()))
    video_path = sample_videos[sample_choice]
    if not os.path.exists(video_path):
        st.error(f"Sample video not found: {video_path}")
        st.stop()
else:
    video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name
    else:
        st.warning("Please upload a video or check 'Use sample video'")
        st.stop()

def extract_pose_landmarks(results):
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    row = []
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])
    return row if len(row) == 33*4 else None

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

def get_point(lm, img_w, img_h):
    return [lm.x * img_w, lm.y * img_h]

# Load LLM once
@st.cache_resource
def load_llm():
    return pipeline("text-generation", model="Qwen/Qwen3-0.6B")

llm = load_llm()

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
predicted_classes = []
angle_records = []

stframe = st.empty()
progress_bar = st.progress(0)

for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    img_h, img_w = frame.shape[:2]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    landmarks = extract_pose_landmarks(results)
    if landmarks:
        input_np = np.array(landmarks).reshape(1, -1)
        pred_class = model.predict(input_np)[0]
        predicted_classes.append(pred_class)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        lm = results.pose_landmarks.landmark
        try:
            left_shoulder = get_point(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value], img_w, img_h)
            left_elbow = get_point(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value], img_w, img_h)
            left_wrist = get_point(lm[mp_pose.PoseLandmark.LEFT_WRIST.value], img_w, img_h)
            left_hip = get_point(lm[mp_pose.PoseLandmark.LEFT_HIP.value], img_w, img_h)
            left_knee = get_point(lm[mp_pose.PoseLandmark.LEFT_KNEE.value], img_w, img_h)
            left_ankle = get_point(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value], img_w, img_h)

            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
            hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

            cv2.putText(frame, f"Elbow: {int(elbow_angle)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            cv2.putText(frame, f"Shoulder: {int(shoulder_angle)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Hip: {int(hip_angle)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Knee: {int(knee_angle)}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            angle_records.append({
                "frame": i,
                "elbow_angle": elbow_angle,
                "shoulder_angle": shoulder_angle,
                "hip_angle": hip_angle,
                "knee_angle": knee_angle,
            })
        except:
            pass

        cv2.putText(frame, f"Predicted: {pred_class}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        predicted_classes.append("NoPose")

    stframe.image(frame, channels="BGR")
    progress_bar.progress((i + 1) / frame_count)

cap.release()

# Results Summary
st.markdown("---")
st.header("Results Summary")

if predicted_classes:
    final_pred = max(set(predicted_classes), key=predicted_classes.count)
    st.success(f"Final predicted class for {exercise_type}: **{final_pred}**")

    st.write("All predictions:")
    st.write(predicted_classes)

    df_pred = pd.DataFrame({"frame": range(len(predicted_classes)), "prediction": predicted_classes})
    st.download_button("Download predictions CSV", df_pred.to_csv(index=False), "predictions.csv", "text/csv")

if angle_records:
    df_angles = pd.DataFrame(angle_records)
    st.write("Joint angles per frame:")
    st.dataframe(df_angles)
    st.download_button("Download angles CSV", df_angles.to_csv(index=False), "angles.csv", "text/csv")

# Prepare prompt for LLM feedback
if predicted_classes and angle_records:
    avg_angles = pd.DataFrame(angle_records).mean(numeric_only=True).to_dict()
    angle_summary = ', '.join([f"{k}: {v:.1f}" for k, v in avg_angles.items()])

    prompt = f"""
ฉันกำลังฝึกท่า {exercise_type} และได้มุมเฉลี่ยดังนี้: {angle_summary}
ผลวิเคราะห์คือ: {"ถูกต้อง" if final_pred == "correct" else "ผิดพลาด"}
ช่วยแนะนำการปรับปรุงท่าทางด้วย
"""

    with st.spinner("Generating AI feedback..."):
        feedback = llm(prompt, max_length=200)[0]['generated_text'].strip()
    st.subheader("🧠 AI Feedback")
    st.text_area("Feedback (Thai LLM):", feedback, height=200)

    # Text to Speech
    def text_to_speech_thai(text, api_key):
        res = requests.get("https://aiforthai.in.th/api/tts", params={
            "text": text,
            "format": "mp3",
            "voice": "f",
            "apikey": api_key
        })
        if res.status_code == 200:
            with open("feedback.mp3", "wb") as f:
                f.write(res.content)
            st.audio("feedback.mp3", format="audio/mp3")
        else:
            st.error("TTS API failed or invalid API key.")

    st.markdown("---")
    st.subheader("🔊 Text to Speech (AIFORTHAI)")

    api_key_input = st.text_input("Enter your AIFORTHAI API key", type="password")
    if st.button("Speak Feedback") and api_key_input.strip() != "":
        text_to_speech_thai(feedback, api_key_input.strip())
