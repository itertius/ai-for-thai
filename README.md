# AI Exercise Form Analysis System 🏋️‍♂️

> An intelligent exercise form analysis system that helps athletes perfect their technique using AI-powered pose detection and real-time feedback in Thai language.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0+-00a393.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8.9+-yellow.svg)](https://mediapipe.dev/)

## 🌟 Features

- **Real-time Form Analysis**
  - Accurate pose detection and tracking
  - Support for multiple exercises (squats, deadlifts, bench press)
  - Instant feedback on form correction
  
- **AI-Powered Coaching**
  - Intelligent chatbot using Gemma 3n
  - Thai language support via VAJA TTS
  - Personalized exercise recommendations

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/itertius/ai-for-thai.git
cd ai-for-thai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run the application**
```bash
python main.py
```

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **AI/ML**: 
  - MediaPipe (Pose Detection)
  - Gemma 3n (Chatbot)
  - VAJA (Thai TTS)
  - Random Forest (Pose Classification)

## 📁 Project Structure

```
├── data/               # Training data and notebooks
│   ├── get_data.ipynb
│   ├── merge.ipynb
│   ├── benchpress/    # Benchpress exercise data
│   ├── deadlift/      # Deadlift exercise data
│   └── squat/         # Squat exercise data
├── models/            # Model training and notebooks
│   ├── classification_angles.ipynb
│   ├── classification.ipynb
│   ├── benchpress/    # Benchpress models
│   ├── deadlift/      # Deadlift models
│   └── squat/         # Squat models
├── deployment/        # Deployment files
│   ├── app.py
│   └── app2.py
├── sample/           # Sample videos
│   ├── benchpress_sample.mp4
│   ├── deadlift_sample.mp4
│   └── squat_sample.mp4
├── main.py          # Main application
├── app3.py         # Additional application file
├── dockerfile     # Docker configuration
└── requirements.txt  # Project dependencies
```

## 🔌 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/predict-pose` | Exercise form analysis |
| `/chat` | AI chatbot interaction |
| `/tts` | Thai text-to-speech |
| `/detect-human` | Human detection |

## 🧪 Testing

```bash
pytest tests/
```

## 📦 Deployment

1. Build Docker image:
```bash
docker build -t exercise-analysis .
```

2. Run container:
```bash
docker run -p 8000:8000 exercise-analysis
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push branch (`git push origin feature/AmazingFeature`)
5. Open a Merge Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- AI for Thai for providing AI services
- MediaPipe for pose detection
- Open source community for their invaluable tools and libraries
