import streamlit as st
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import cv2
import tempfile
import os
from dotenv import load_dotenv
import requests

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("AIFORTHAI_API_KEY")  # สำหรับ Pathumma LLM
tts_api_key = os.getenv("TTS_API_KEY")    # สำหรับ Vaja TTS

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
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not enable_camera else 100
predicted_classes = []

stframe = st.empty()
progress_bar = st.progress(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    landmarks = extract_pose_landmarks(results)
    if landmarks:
        input_np = np.array(landmarks).reshape(1, -1)
        pred_class = model.predict(input_np)[0]
        predicted_classes.append(pred_class)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, f"Predicted: {pred_class}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    stframe.image(frame, channels="BGR")

    frame_count += 1
    if not enable_camera:
        progress = int((frame_count / frame_total) * 100)
        progress_bar.progress(min(progress, 100))

    if enable_camera and frame_count >= 100:
        break

cap.release()

# --- Label Thai Map ---
label_thai_map = {
    "b_correct_up": "เบนช์เพรส ช่วงขึ้นถูกต้อง",
    "b_correct_down": "เบนช์เพรส ช่วงลงถูกต้อง",
    "b_excessive_arch_up": "เบนช์เพรส หลังโก่งเกินไปช่วงขึ้น",
    "b_excessive_arch_down": "เบนช์เพรส หลังโก่งเกินไปช่วงลง",
    "b_arms_spread_up": "เบนช์เพรส แขนกางช่วงขึ้น",
    "b_arms_spread_down": "เบนช์เพรส แขนกางช่วงลง",
    "d_correct_up": "เดดลิฟต์ ช่วงขึ้นถูกต้อง",
    "d_correct_down": "เดดลิฟต์ ช่วงลงถูกต้อง",
    "d_spine_neutral_up": "เดดลิฟต์ หลังตรงช่วงขึ้น",
    "d_spine_neutral_down": "เดดลิฟต์ หลังตรงช่วงลง",
    "d_arms_spread_up": "เดดลิฟต์ แขนกางช่วงขึ้น",
    "d_arms_spread_down": "เดดลิฟต์ แขนกางช่วงลง",
    "d_arms_narrow_up": "เดดลิฟต์ แขนแคบช่วงขึ้น",
    "d_arms_narrow_down": "เดดลิฟต์ แขนแคบช่วงลง",
    "s_correct_up": "สควอท ช่วงขึ้นถูกต้อง",
    "s_correct_down": "สควอท ช่วงลงถูกต้อง",
    "s_spine_neutral_up": "สควอท หลังตรงช่วงขึ้น",
    "s_spine_neutral_down": "สควอท หลังตรงช่วงลง",
    "s_caved_in_knees_up": "สควอท เข่าบิดเข้าช่วงขึ้น",
    "s_caved_in_knees_down": "สควอท เข่าบิดเข้าช่วงลง",
    "s_feet_spread_up": "สควอท เท้ากางช่วงขึ้น",
    "s_feet_spread_down": "สควอท เท้ากางช่วงลง",
}

# --- Text-to-Speech function using AIFORTHAI Vaja API ---
def play_tts(text, api_key):
    url = "https://api.aiforthai.in.th/vaja"
    headers = {
        "Apikey": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "speaker": "nana"
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        resp_json = response.json()
        audio_url = resp_json.get("audio_url")
        if audio_url:
            audio_resp = requests.get(audio_url)
            if audio_resp.status_code == 200:
                st.audio(audio_resp.content, format="audio/wav")
            else:
                st.warning(f"ไม่สามารถดาวน์โหลดไฟล์เสียงได้: {audio_resp.status_code}")
        else:
            st.warning(f"ไม่พบ audio_url ใน response: {resp_json}")
    else:
        st.warning(f"❌ ไม่สามารถสังเคราะห์เสียงได้: {response.status_code} - {response.text}")


# --- Display Final Results and LLM feedback ---
if predicted_classes:
    final_class = max(set(predicted_classes), key=predicted_classes.count)
    st.success(f"✅ Final Prediction for {exercise_type}: **{final_class}**")

    st.write("🧮 Frame-by-frame predictions:")
    st.write(predicted_classes)

    df = pd.DataFrame({
        "Frame": range(len(predicted_classes)),
        "Prediction": predicted_classes
    })
    st.download_button("📥 Download Predictions as CSV", df.to_csv(index=False),
                       file_name="predictions.csv", mime="text/csv")

    # Import Pathumma LLM (aift package)
    from aift.multimodal import textqa
    from aift import setting

    setting.set_api_key(api_key)

    wrong_keywords = [
        "excessive", "spread", "spine_neutral",
        "arms_spread", "narrow", "caved_in", "feet_spread"
    ]

    st.subheader("🤖 คำแนะนำจาก Pathumma LLM แบบรายวินาที")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    shown_seconds = set()

    for i, label in enumerate(predicted_classes):
        label_lower = label.lower()
        if any(k in label_lower for k in wrong_keywords):
            second = int(i / fps)
            if second in shown_seconds:
                continue
            shown_seconds.add(second)

            label_th = label_thai_map.get(label, label)
            prompt = f"ให้คำแนะนำเกี่ยวกับท่า {exercise_type} โดยพิจารณาจาก label ว่า {label_th}"

            try:
                response = textqa.generate(prompt, return_json=True)
                suggestion = response.get("content", "ไม่มีคำแนะนำ")
                st.markdown(f"🕐 **วินาทีที่ {second}** — {suggestion}")
                play_tts(suggestion, tts_api_key)
            except Exception as e:
                st.warning(f"⚠️ เกิดข้อผิดพลาดในวินาทีที่ {second}: {e}")

    if not shown_seconds:
        st.success("👏 ไม่พบท่าผิดที่ต้องแนะนำเพิ่มเติม")
        play_tts("คุณทำได้ดีมากครับ พยายามต่อไปนะครับ", tts_api_key)
