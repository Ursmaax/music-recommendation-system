# 🌌 Music Recommendation System
### *Where Emotions Meet Melody*

## 🚀 Run the Application Online

You can run the full **Music Recommendation System** directly in your browser — no installation needed.

👉 **Try it on Hugging Face Spaces:**  
https://huggingface.co/spaces/umamaheshsativada/music-recommendation-system

[![Run on Hugging Face Spaces](https://img.shields.io/badge/HuggingFace-Spaces-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/umamaheshsativada/music-recommendation-system)


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![Spotify](https://img.shields.io/badge/Spotify-API-1DB954?style=for-the-badge&logo=spotify&logoColor=white)](https://developer.spotify.com/)

---

## 📖 Executive Summary
The **Music Recommendation System** is a cutting-edge AI application designed to bridge the gap between human emotion and musical experience. By leveraging advanced computer vision and deep learning techniques, the system analyzes real-time video feed to detect the user's emotional state and instantly curates a Spotify playlist that resonates with their mood.

Whether you're feeling happy, sad, energetic, or calm, this system ensures your environment is always in sync with your feelings. Wrapped in a stunning "Dreamland" aesthetic interface, it offers a seamless and immersive user experience.

---

## ✨ Key Features

*   **🎭 Real-Time Emotion Recognition:** Utilizes a high-performance Fusion Model (combining MobileNet and Custom CNN) to detect 7 distinct emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) with high accuracy.
*   **🎵 Intelligent Music Curation:** dynamically interfaces with the Spotify API to fetch tracks that mathematically match the energy and valence of the detected emotion.
*   **👁️ Computer Vision Powered:** Processes live webcam feed using OpenCV and TensorFlow for instant feedback.
*   **🎨 Dreamland UI:** A fully custom-designed user interface featuring glassmorphism, aurora gradients, and smooth animations for a premium feel.
*   **🚀 Cloud-Native Architecture:** Optimized for deployment on Hugging Face Spaces with Docker-based containerization.

---

## 🛠️ Technology Stack

*   **Core Language:** Python 3.10+
*   **Deep Learning:** TensorFlow, Keras
*   **Computer Vision:** OpenCV (cv2)
*   **Web Framework:** Gradio (with custom CSS/JS)
*   **Audio Processing:** Librosa
*   **API Integration:** Spotipy (Spotify Web API)

---

## 📂 Project Structure

```
Music_Rec_System/
├── app.py                  # Main application entry point (Gradio)
├── requirements.txt        # Project dependencies
├── README.md               # Documentation
├── assets/                 # Static assets (CSS, JS, Images)
│   ├── css/custom_ui.css   # Custom styling
│   └── js/                 # UI interactions
├── models/                 # Pre-trained AI models
│   ├── fusion_model.h5     # The core emotion detection model
│   └── labels.json         # Emotion label mappings
└── src/                    # Source code
    ├── fusion_inference.py # Inference logic
    ├── spotify_auth.py     # Spotify API handling
    └── preprocess.py       # Data preprocessing utilities
```

---

## 🚀 Installation & Setup

Follow these steps to set up the project locally.

### 1. Prerequisites
*   Python 3.10 or higher installed.
*   A Spotify Developer account (to get Client ID and Secret).

### 2. Clone the Repository
```bash
git clone https://github.com/Ursmaax/music-recommendation-system.git
cd music-recommendation-system
```

### 3. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 4. Configure Spotify Credentials
Create a `.env` file or set environment variables for your Spotify API keys:
```bash
export SPOTIPY_CLIENT_ID='your_client_id'
export SPOTIPY_CLIENT_SECRET='your_client_secret'
```

### 5. Run the Application
```bash
python app.py
```
The application will launch at `http://localhost:7860`.



---

## 🧠 Model Architecture

The system uses a **Fusion Model** approach for robustness:
1.  **Input:** Live video frames from the webcam.
2.  **Preprocessing:** Face detection using Haar Cascades, followed by normalization.
3.  **Feature Extraction:** 
    *   **Branch A:** MobileNetV2 (Transfer Learning) for high-level feature extraction.
    *   **Branch B:** Custom CNN for fine-grained expression analysis.
4.  **Fusion:** The outputs are concatenated and passed through dense layers to predict the final emotion probability.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

*Built with ❤️ by [Ursmaax]*
