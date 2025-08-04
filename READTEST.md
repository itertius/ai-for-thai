ได้ครับ! นี่คือ **step-by-step guide** สำหรับการทดสอบ `curl` กับ Docker Compose config ที่คุณให้มา โดยมี 2 services คือ `pose-api` รันบนพอร์ต 3401 และ `ollama` รันบนพอร์ต 3402

---

## Step 1: ตรวจสอบว่า Docker Compose รันอยู่

```bash
docker-compose up -d
```

รอจน container ทั้งสองรันขึ้นมา (เช็คด้วย `docker ps`)

---

## Step 2: ตรวจสอบว่าแต่ละ service เปิดพอร์ตถูกต้อง

* `pose-api` เปิดบน `localhost:3401`
* `ollama` เปิดบน `localhost:3402`

ลองเช็คด้วยคำสั่ง:

```bash
curl http://localhost:3401/health
curl http://localhost:3402
```

(ถ้าไม่มี `/health` ลองดูเอกสาร API หรือ `/docs` สำหรับ pose-api)

---

## Step 3: ทดสอบเรียก API `/predict-pose` (pose-api)

### ตัวอย่าง curl

สมมติคุณมีไฟล์วิดีโอ `sample/squat_sample.mp4` และต้องการส่งไปพร้อมพารามิเตอร์ `exercise=squat`

```bash
curl -X POST "http://localhost:3401/predict-pose" \
  -H "accept: application/json" \
  -F "exercise=squat" \
  -F "video=@sample/squat_sample.mp4"
```

* `-F` คือส่ง multipart form-data
* `exercise` ต้องระบุชื่อท่าทางให้ตรงกับโมเดลในระบบ
* `video` คือไฟล์วิดีโอที่ส่งไป

---

## Step 4: ทดสอบเรียก API `/chat` (pose-api ที่ทำหน้าที่เรียก ollama)

```bash
  curl -X POST "http://localhost:3401/chat" \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello, how are you?"}'
```

ถ้าคอนฟิกถูกต้อง จะได้ผลลัพธ์จาก ollama model ที่กำหนด (เช่น qwen3:0.6b)

---

## Step 5: ตรวจสอบข้อผิดพลาด

* ถ้าเจอ error เช่น `connection refused` หรือ `404` ให้เช็คว่า container รันอยู่จริงหรือไม่
* ถ้าได้ error `422 Unprocessable Entity` แสดงว่า parameter ไม่ครบ หรือชื่อพารามิเตอร์ไม่ตรงกับ FastAPI function signature
* ถ้าเจอ error `Ollama API error: 404` ให้ตรวจสอบ URL ollama และพอร์ตใน environment variable `OLLAMA_API_URL`

---

## Step 6: ตัวอย่าง response ที่คาดหวัง

* `/predict-pose`

```json
{
  "exercise": "squat",
  "prediction": 1
}
```

* `/chat`

```json
{
  "id": "...",
  "object": "chat.completion",
  "created": 1234567890,
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      }
    }
  ]
}
```

---

## สรุปคำสั่ง curl ที่ใช้งานบ่อย

```bash
# ทดสอบ pose prediction
curl -X POST "http://localhost:3401/predict-pose" \
  -H "accept: application/json" \
  -F "exercise=squat" \
  -F "video=@sample/squat_sample.mp4"

# ทดสอบ chat
curl -X POST "http://localhost:3401/chat" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?"}'
```

---