# Aura Guard Security Gate 🛡️

A state-of-the-art Hands-Free Face Recognition check-in system built with Python, Flask, and OpenCV's YuNet/SFace models.

## Features ✨
- **Ultra-fast Face Detection:** Powered by OpenCV YuNet.
- **High-Accuracy Recognition:** Uses SFace for facial embeddings and cosine similarity matching.
- **Hands-Free Mode:** Continuously scans and automatically unlocks when authorized personnel approach.
- **Beautiful UI:** A stunning, premium glassmorphism dashboard with live camera feeds and security logs.
- **Offline Capable:** Runs entirely locally without needing internet connection for the AI processing.

## Installation 🚀
1. Clone this repository
2. Install the required Python packages:
   ```bash
   pip install flask flask-cors opencv-python numpy fpdf
   ```
3. Run the Flask Server:
   ```bash
   python app.py
   ```
4. Open your browser to `http://localhost:5000`

## Technologies Used 💻
- **Backend:** Python, Flask, SQLite
- **Frontend:** Vanilla JS, CSS3 (Glassmorphism design), HTML5
- **AI Models:** OpenCV YuNet & SFace ONNX models
