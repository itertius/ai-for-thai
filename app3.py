import streamlit as st
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import os
import cv2
import requests

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

st.title("🏋️ Powerlifting Pose Classification with AI Feedback & LLM Advice")

exercise_type = st.selectbox("Select Exercise Type", ["benchpress", "deadlift", "squat"])

# Input source selection
input_source = st.radio("Select video input source", ("Upload/sample video", "Use camera"))

if input_source == "Use camera":
    cam_file = st.camera_input("Record a short video or take a photo")

    if cam_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(cam_file.read())
        video_path = tfile.name
    else:
        st.warning("Please record a video or take a photo with your camera")
        st.stop()
else:
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
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            video_path = tfile.name
        else:
            st.warning("Please upload a video or check 'Use sample video'")
            st.stop()

# Pose prediction API call
def call_pose_api_multiform(frame, pose_type):
    url = "https://api.hackathon2025.ai.in.th/team24-2/predict-pose"
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {
        "file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg"),
    }
    data = {
        "pose_type": pose_type
    }
    try:
        resp = requests.post(url, files=files, data=data, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"Pose API call failed: {e}")
        return None
    except Exception as e:
        st.error(f"Pose API call error: {e}")
        return None

# LLM advice API call
def call_llm_advice(label):
    url = "https://api.hackathon2025.ai.in.th/team24-4/advise"
    headers = {"Content-Type": "application/json"}
    payload = {"label": label}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        st.write("🔁 LLM Response:", data)  # Debug line
        return data.get("reply", "No reply from LLM")
    except requests.exceptions.HTTPError as e:
        st.error(f"⚠️ LLM API HTTP error: {e}")
        content = resp.text if 'resp' in locals() else 'No response'
        st.error(f"Response content: {content}")
        return "LLM API failed."
    except Exception as e:
        st.error(f"⚠️ LLM API error: {e}")
        return "LLM API failed."

# Open video and process frames
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

predicted_classes = []

stframe = st.empty()
progress_bar = st.progress(0)

for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Predict every 30 frames
    if i % 60 == 0:
        api_res = call_pose_api_multiform(frame, exercise_type)
        if api_res:
            pred_label = api_res.get("label", "Unknown")
        else:
            pred_label = "API_Error"
        st.write(f"Second {round(i/60)}: {pred_label}")
    else:
        pred_label = "Skipped"

    predicted_classes.append(pred_label)

    # Show prediction label
    if pred_label not in ["Skipped", "API_Error"]:
        cv2.putText(frame, f"Prediction: {pred_label}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Stream to UI
    stframe.image(frame, channels="BGR")
    progress_bar.progress((i + 1) / frame_count)

cap.release()

# Summary
st.markdown("---")
st.header("Prediction Summary")

# Filter out invalid and unhelpful labels for advice
valid_preds = [
    p for p in predicted_classes
    if p not in ("Skipped", "API_Error", "no_pose_detected")
]

if valid_preds:
    final_pred = max(set(valid_preds), key=valid_preds.count)
else:
    final_pred = "No valid prediction"

st.success(f"Final predicted class for {exercise_type}: **{final_pred}**")

df_pred = pd.DataFrame({
    "frame": range(len(predicted_classes)),
    "prediction": predicted_classes
})
st.write(df_pred)
st.download_button("📥 Download predictions CSV", df_pred.to_csv(index=False), "predictions.csv", "text/csv")

# Mapping your predicted labels to the API’s VALID_LABELS
label_map = {
    "d_correct_down": "correct_deadlift",
    "d_correct_up": "correct_deadlift",
    "d_incorrect_rounded_back": "incorrect_deadlift_rounded_back",
    "d_incorrect_hips_rise_first": "incorrect_deadlift_hips_rise_first",
    "s_correct": "correct_squat",
    "s_incorrect_back": "incorrect_squat_back",
    "s_incorrect_knees_in": "incorrect_squat_knees_in",
    "b_correct": "correct_benchpress",
    "b_incorrect_flared_elbows": "incorrect_benchpress_flared_elbows",
    "b_incorrect_wrist": "incorrect_benchpress_wrist_position",
    # Add more mappings if needed
}

if final_pred != "No valid prediction":
    mapped_label = label_map.get(final_pred)
    if not mapped_label:
        st.info(f"No valid advice available for label '{final_pred}'.")
    else:
        with st.spinner("🧠 Getting advice from LLM..."):
            advice = call_llm_advice(mapped_label)
        st.subheader("🧠 AI Advice")
        st.write(advice)
else:
    st.info("No valid pose predictions to get advice.")