## Overview

โปรเจกต์นี้เป็นระบบวิเคราะห์ท่าทางการออกกำลังกายที่ใช้ **MediaPipe** เพื่อดึง key points จากร่างกาย และฝึกโมเดล **Random Forest** สำหรับจำแนกความถูกต้องของท่าทาง พร้อมด้วยระบบ **Chatbot** สำหรับให้คำแนะนำผ่านโมเดล **Gemma 3n (Ollama)** และฟีเจอร์แปลงข้อความเป็นเสียงภาษาไทยด้วยโมเดล **VAJA จาก AIFORTHAI**

---

## Objectives

* วิเคราะห์ท่าทางการออกกำลังกายจากวิดีโอหรือข้อมูล key points
* ประเมินความถูกต้องของท่าทางด้วย Machine Learning
* ให้คำแนะนำและตอบคำถามผู้ใช้ผ่านระบบ Chatbot (Gemma 3n)
* แปลงคำแนะนำจากข้อความเป็นเสียงด้วยระบบ TTS ภาษาไทย (VAJA)

---

## Challenges

* การเก็บและเตรียมข้อมูล key points ที่แม่นยำจากวิดีโอจริง
* การฝึกโมเดล Random Forest ให้ครอบคลุมผู้ใช้ที่มีความหลากหลายของท่าทาง
* การผสานระบบ LLM (Gemma 3n) และ TTS (AIFORTHAI) เข้ากับระบบวิเคราะห์ท่าทาง
* ปรับปรุงประสิทธิภาพและลด latency เพื่อรองรับการตอบสนองแบบเรียลไทม์

---

## API Endpoints

### 1. `/predict-pose`

* **Method:** POST
* **Input:** JSON ที่ประกอบด้วย key points จาก MediaPipe
* **Process:**
  1. รับข้อมูล key points จากผู้ใช้
  2. ประมวลผลด้วยโมเดล Random Forest
* **Output:** ผลการประเมินท่าทาง เช่น `"Correct"` หรือ `"Incorrect"` พร้อม confidence score

---

### 2. `/chat`

* **Method:** POST
* **Input:** JSON ที่มีข้อความคำถามจากผู้ใช้
* **Process:**
  1. รับข้อความจากผู้ใช้
  2. ส่งข้อความไปยังโมเดล `Gemma 3n` ผ่าน Ollama
* **Output:** คำตอบหรือคำแนะนำจากโมเดล

---

### 3. `/aiforthai-tts`

* **Method:** POST
* **Input:** JSON ที่มีข้อความภาษาไทย
* **Process:**
  1. ส่งข้อความไปยัง AIFORTHAI Text-to-Speech API
  2. แปลงข้อความเป็นเสียง
* **Output:** ไฟล์เสียง `.wav` หรือ `.mp3` หรือ URL สำหรับดาวน์โหลดเสียง

---

### 4. `/aiforthai-pathumma-vision`
* **Method:** POST
* **Input:** รูปภาพ
* **Process:**
  1. ส่งรูปภาพไปยัง AIFORTHAI API เพื่อวิเคราะห์และถามคำถามเพิ่มเติม
  2. แปลงข้อความเป็นเสียง
* **Output:** คำตอบที่ได้จาก LLM


---

### 5. `/aiforthai-human-detect`
* **Method:** POST
* **Input:** รูปภาพ
* **Process:**
  1. ส่งรูปภาพเข้า AIFORTHAI API เพือวิเคราะ์ลักษณะเฉพาะบุคคล
  2. แปลงข้อความเป็นเสียง
* **Output:** ลักษณะเฉพาะตัวของแต่ละบุคคล

---

## Port Mappings

| Service             | Container Port | Host Port |
| ------------------- | -------------- | --------- |
| Pose Prediction API | 3401           | 3401      |
| Chatbot API         | 3402           | 3402      |
| AIFORTHAI TTS API   | 3403           | 3403      |