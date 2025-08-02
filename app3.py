import streamlit as st
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import tempfile
import os
import cv2
import requests
import json

# ---- Initialize MediaPipe Pose ----
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# ---- UI ----
st.title("🏋️ Powerlifting Pose Classification with AI Feedback & TTS")

exercise_type = st.selectbox("Select Exercise Type", ["Benchpress", "Deadlift", "Squat"])

# Model files paths
model_paths = {
    "Benchpress": "models/benchpress/benchpress.pkl",
    "Deadlift": "models/deadlift/deadlift.pkl",
    "Squat": "models/squat/squat.pkl"
}

# Load model safely with error handling
try:
    with open(model_paths[exercise_type], "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Model file not found for {exercise_type}. Please check the path.")
    st.stop()

# Video selection
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

# ---- Helper functions ----
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

# ---- Load LLM function (via Ollama API) ----
@st.cache_resource
def load_llm():
    def local_llm(prompt, max_length=200):
        url = "http://host.docker.internal:11434/api/generate"  # Your LLM API URL
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "qwen3:0.6b",
            "prompt": prompt,
            "options": {"num_predict": max_length}
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
            
            raw_text = response.text.strip()
            # Debug print raw response to help troubleshoot JSON issues
            print("LLM API raw response:", raw_text)
            
            try:
                # Try normal JSON parsing
                result = response.json()
            except json.JSONDecodeError as e:
                # If failed, try parsing first line only (sometimes server sends multiple JSONs)
                print(f"JSONDecodeError: {e} — trying to parse first line only")
                first_line = raw_text.split('\n')[0]
                result = json.loads(first_line)
            
            # Return the text from "response" key or fallback error
            return result.get("response", "⚠️ LLM Error: no 'response' field in JSON")
        
        except requests.exceptions.RequestException as e:
            return f"⚠️ LLM call failed: {e}"
        except Exception as e:
            return f"⚠️ Unexpected error: {e}"
    
    return local_llm

llm = load_llm()

# ---- Process video frames ----
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

            # Overlay angles on frame
            for j, (label, value) in enumerate({
                "Elbow": elbow_angle,
                "Shoulder": shoulder_angle,
                "Hip": hip_angle,
                "Knee": knee_angle
            }.items()):
                cv2.putText(frame, f"{label}: {int(value)}", (10, 30 + j * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255 - j * 50, j * 60), 2)

            angle_records.append({
                "frame": i,
                "elbow_angle": elbow_angle,
                "shoulder_angle": shoulder_angle,
                "hip_angle": hip_angle,
                "knee_angle": knee_angle,
            })
        except Exception:
            # silently skip angle calc errors
            pass

        cv2.putText(frame, f"Predicted: {pred_class}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        predicted_classes.append("NoPose")

    stframe.image(frame, channels="BGR")
    progress_bar.progress((i + 1) / frame_count)

cap.release()

# ---- Show results ----
st.markdown("---")
st.header("Results Summary")

if predicted_classes:
    final_pred = max(set(predicted_classes), key=predicted_classes.count)
    st.success(f"Final predicted class for {exercise_type}: **{final_pred}**")

    df_pred = pd.DataFrame({"frame": range(len(predicted_classes)), "prediction": predicted_classes})
    st.write(df_pred)
    st.download_button("📥 Download predictions CSV", df_pred.to_csv(index=False), "predictions.csv", "text/csv")

if angle_records:
    df_angles = pd.DataFrame(angle_records)
    st.write("Joint angles per frame:")
    st.dataframe(df_angles)
    st.download_button("📥 Download angles CSV", df_angles.to_csv(index=False), "angles.csv", "text/csv")

# ---- AI Feedback via LLM ----
if predicted_classes and angle_records:
    avg_angles = pd.DataFrame(angle_records).mean(numeric_only=True).to_dict()
    angle_summary = ', '.join([f"{k}: {v:.1f}" for k, v in avg_angles.items()])
    prompt = f"""
ฉันกำลังฝึกท่า {exercise_type} และได้มุมเฉลี่ยดังนี้: {angle_summary}
ผลวิเคราะห์คือ: {"ถูกต้อง" if final_pred == "correct" else "ผิดพลาด"}
ช่วยแนะนำการปรับปรุงท่าทางด้วย
"""

    with st.spinner("🧠 Generating AI feedback..."):
        feedback = llm(prompt, max_length=200)

    st.subheader("🧠 AI Feedback")
    st.text_area("Feedback :", feedback, height=200)

    # ---- Text-to-Speech (AIFORTHAI) ----
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
            st.error("⚠️ TTS API failed or invalid API key.")

    st.markdown("---")
    st.subheader("🔊 Text to Speech (AIFORTHAI)")
    api_key_input = st.text_input("Enter your AIFORTHAI API key", type="password")
    if st.button("Speak Feedback") and api_key_input.strip():
        text_to_speech_thai(feedback, api_key_input.strip())