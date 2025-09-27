# 🧘 Yogalyze – AI Yoga Trainer

**Yogalyze** is a real-time AI-powered yoga trainer built with **Flask**, **OpenCV**, **MediaPipe**, and a **trained machine learning model**.  
It detects yoga poses via your webcam, provides **instant visual and audio feedback**, and helps users improve their yoga practice through guided sessions and pose tracking.  

---

## 🌟 Motivation

Practicing yoga requires proper posture and timing. Incorrect poses can lead to injuries or reduced benefits. Yogalyze solves this by:  

- Detecting user poses in real-time  
- Providing feedback for incorrect posture  
- Guiding users through yoga flows with voice and visual cues  
- Allowing users to track performance and progress  

This project combines **Computer Vision**, **Machine Learning**, and **Web Development** to make yoga accessible, interactive, and engaging at home or on-the-go.  

---

## ✨ Features

- 📸 **Real-time Pose Detection**: Uses webcam feed to track body landmarks with MediaPipe  
- 🧠 **ML-Powered Pose Classification**: A trained ML model verifies the user’s yoga poses  
- 🗣️ **Audio Instructions**: Voice guidance for pose corrections and session prompts using `pyttsx3`  
- ✅ **Visual Feedback**: Highlights correct and incorrect poses with on-screen indicators  
- 📊 **Session Tracking**: Tracks pose duration, hold times, and sequence completion  
- 🔄 **Multi-Pose Sequence**: Supports full yoga routines with adjustable hold times per pose  
- 🌐 **Flask Web Interface**: Lightweight, CORS-enabled backend for frontend integration  

---

## 📦 Installation & Requirements

### Prerequisites
- Python **3.7+**  
- `pip` (Python package manager)  

### Clone the Repository
```bash
git clone https://github.com/your-username/yogalyze.git
cd yogalyze
