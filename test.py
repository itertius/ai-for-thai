import requests

url = "https://api.hackathon2025.ai.in.th/team24-4/advise"  # เปลี่ยนเป็น URL ที่รัน FastAPI จริง
payload = {"label": "correct_squat"}
headers = {"Content-Type": "application/json"}

resp = requests.post(url, json=payload, headers=headers)
print("Status code:", resp.status_code)
print("Response:", resp.json())
