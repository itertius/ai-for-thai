import os
import pickle
import numpy as np
import tempfile
import requests
import cv2
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import logging

app = FastAPI()

logger = logging.getLogger("uvicorn.error")

# Load models on startup
model_paths: Dict[str, str] = {
    "squat": "models/squat/squat.pkl",
    "deadlift": "models/deadlift/deadlift.pkl",
    "benchpress": "models/benchpress/benchpress.pkl"
}

models: Dict[str, object] = {}

@app.on_event("startup")
def load_models():
    for name, path in model_paths.items():
        try:
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
            logger.info(f"Loaded model '{name}' from '{path}'")
        except Exception as e:
            logger.warning(f"Failed to load model '{name}': {e}")

# Environment variables
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
AIFORTHAI_API_KEY = os.getenv("AIFORTHAI_API_KEY", "")

mp_pose = mp.solutions.pose

def extract_mean_keypoints_from_video(video_path: str) -> Optional[np.ndarray]:
    pose = mp_pose.Pose(static_image_mode=False)
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            frame_kp = []
            for lm in results.pose_landmarks.landmark:
                frame_kp.extend([lm.x, lm.y, lm.z, lm.visibility])
            keypoints_list.append(frame_kp)

    cap.release()
    pose.close()

    if not keypoints_list:
        return None

    mean_keypoints = np.mean(np.array(keypoints_list), axis=0)
    return mean_keypoints.reshape(1, -1)

@app.post("/predict-pose")
async def predict_pose(exercise: str = Form(...), video: UploadFile = File(...)):
    if exercise not in models:
        raise HTTPException(status_code=400, detail=f"Unsupported exercise '{exercise}'. Available: {list(models.keys())}")

    tmp_path = None
    try:
        # Save uploaded video temporarily
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_path = tmp.name
        tmp.write(await video.read())
        tmp.close()

        features = extract_mean_keypoints_from_video(tmp_path)
        if features is None:
            raise HTTPException(status_code=400, detail="No pose landmarks detected in the video.")

        model = models[exercise]

        # Prediction and confidence
        proba = model.predict_proba(features)
        pred = model.predict(features)[0]

        return {"prediction": str(pred), "confidence": float(proba.max())}

    except Exception as e:
        logger.error(f"Error in predict_pose: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to predict pose: {e}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# Chat API to Ollama LLM
class ChatPrompt(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(prompt: ChatPrompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt.prompt}]
    }
    try:
        response = requests.post(f"{OLLAMA_API_URL}/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama API error: {e}")

# Text-to-Speech API using AIFORTHAI
class TTSPrompt(BaseModel):
    text: str
    speaker: str = "1"

@app.post("/aiforthai-tts")
async def aiforthai_tts(req: TTSPrompt):
    headers = {
        "Apikey": AIFORTHAI_API_KEY,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "text": req.text,
        "speaker": req.speaker
    }
    url = "https://api.aiforthai.in.th/vaja9/synth_audiovis"

    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"AIFORTHAI TTS API error: {e}")
        raise HTTPException(status_code=500, detail=f"AIFORTHAI TTS API error: {e}")

# Human segmentation API
@app.post("/aiforthai-human-detect")
async def aiforthai_human_detect(image: UploadFile = File(...)):
    headers = {"Apikey": AIFORTHAI_API_KEY}
    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp_path = tmp.name
        tmp.write(await image.read())
        tmp.close()

        with open(tmp_path, "rb") as f:
            files = {"image": f}
            response = requests.post("https://api.aiforthai.in.th/humanseg", headers=headers, files=files)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"AIFORTHAI Human Detect API error: {e}")
        raise HTTPException(status_code=500, detail=f"AIFORTHAI Human Detect API error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# Pathumma Vision API
@app.post("/aiforthai-pathumma-vision")
async def aiforthai_pathumma_vision(image: UploadFile = File(...)):
    headers = {"Apikey": AIFORTHAI_API_KEY}
    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp_path = tmp.name
        tmp.write(await image.read())
        tmp.close()

        with open(tmp_path, "rb") as f:
            files = {"file": f}
            response = requests.post("https://api.aiforthai.in.th/pathummavision", headers=headers, files=files)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"AIFORTHAI Pathumma Vision API error: {e}")
        raise HTTPException(status_code=500, detail=f"AIFORTHAI Pathumma Vision API error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
