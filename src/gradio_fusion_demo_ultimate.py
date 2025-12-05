"""
🌌 Music Recommendation System
World-class emotion detection with dreamland aesthetics
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import time

from fusion_inference import predict_from_live, get_emotion_color, get_emotion_description
from spotify_auth import spotify_client

# 🎨 WORLD-CLASS DREAMLAND CSS
DREAMLAND_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Playfair+Display:wght@700;900&display=swap');

:root {
    --aurora-pink: #ff6ec7;
    --aurora-purple: #a78bfa;
    --aurora-blue: #60a5fa;
    --aurora-cyan: #06b6d4;
    --aurora-green: #10b981;
    --dream-bg: #0a0e27;
    --dream-card: rgba(20, 25, 54, 0.7);
    --glow: rgba(167, 139, 250, 0.5);
}

* {
    font-family: 'Poppins', sans-serif !important;
}

body {
    background: var(--dream-bg) !important;
    overflow-x: hidden;
}

/* 🌟 Animated Aurora Background */
.gradio-container {
    position: relative;
    max-width: 1600px !important;
    margin: 0 auto !important;
}

.gradio-container::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        radial-gradient(circle at 20% 20%, var(--aurora-pink) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, var(--aurora-blue) 0%, transparent 50%),
        radial-gradient(circle at 50% 80%, var(--aurora-purple) 0%, transparent 50%),
        radial-gradient(circle at 10% 60%, var(--aurora-cyan) 0%, transparent 50%);
    opacity: 0.15;
    animation: aurora 20s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes aurora {
    0%, 100% { transform: scale(1) rotate(0deg); opacity: 0.15; }
    33% { transform: scale(1.1) rotate(5deg); opacity: 0.2; }
    66% { transform: scale(0.95) rotate(-5deg); opacity: 0.25; }
}

/* ✨ Floating Particles */
.gradio-container::after {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(2px 2px at 20% 30%, white, transparent),
        radial-gradient(2px 2px at 60% 70%, white, transparent),
        radial-gradient(1px 1px at 50% 50%, white, transparent),
        radial-gradient(1px 1px at 80% 10%, white, transparent),
        radial-gradient(2px 2px at 90% 60%, white, transparent),
        radial-gradient(1px 1px at 33% 80%, white, transparent);
    background-size: 200% 200%;
    animation: particles 30s linear infinite;
    opacity: 0.4;
    pointer-events: none;
    z-index: 0;
}

@keyframes particles {
    0% { transform: translateY(0); }
    100% { transform: translateY(-100%); }
}

/* 🎭 Main Title - Gorgeous */
h1 {
    background: linear-gradient(135deg, var(--aurora-pink), var(--aurora-purple), var(--aurora-blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Playfair Display', serif !important;
    font-weight: 900 !important;
    font-size: 4rem !important;
    text-align: center;
    margin: 2rem 0 !important;
    animation: titleGlow 3s ease-in-out infinite;
    letter-spacing: -2px;
    position: relative;
    z-index: 1;
}

@keyframes titleGlow {
    0%, 100% { filter: drop-shadow(0 0 20px var(--glow)); }
    50% { filter: drop-shadow(0 0 40px var(--aurora-purple)); }
}

h3 {
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.3rem !important;
    margin: 1rem 0 !important;
    position: relative;
    z-index: 1;
}

/* 🎨 Premium Input Cards */
.gr-box {
    background: var(--dream-card) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(167, 139, 250, 0.2) !important;
    border-radius: 24px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 60px rgba(167, 139, 250, 0.1) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
    z-index: 1;
}

.gr-box:hover {
    transform: translateY(-5px);
    border-color: var(--aurora-purple) !important;
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.4), 0 0 80px rgba(167, 139, 250, 0.3) !important;
}

/* 🎤 File Upload Areas - Dreamy */
.gr-file,
.gr-audio {
    background: linear-gradient(135deg, rgba(167, 139, 250, 0.1), rgba(96, 165, 250, 0.1)) !important;
    border: 2px dashed var(--aurora-purple) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    transition: all 0.3s ease !important;
}

.gr-file:hover,
.gr-audio:hover {
    border-color: var(--aurora-pink) !important;
    background: linear-gradient(135deg, rgba(255, 110, 199, 0.15), rgba(167, 139, 250, 0.15)) !important;
    transform: scale(1.02);
}

/* 🚀 Predict Button - GORGEOUS */
.gr-button-primary {
    background: linear-gradient(135deg, #ff6ec7, #a78bfa, #60a5fa) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.4rem !important;
    padding: 1.2rem 3rem !important;
    border-radius: 50px !important;
    box-shadow: 0 10px 40px rgba(255, 110, 199, 0.4), 0 0 80px rgba(167, 139, 250, 0.3) !important;
    transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55) !important;
    position: relative;
    overflow: hidden;
    z-index: 1;
}

.gr-button-primary::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(255,255,255,0.3), transparent);
    transform: rotate(45deg);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
    100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
}

.gr-button-primary:hover {
    transform: translateY(-8px) scale(1.05) !important;
    box-shadow: 0 15px 60px rgba(255, 110, 199, 0.6), 0 0 100px rgba(167, 139, 250, 0.5) !important;
}

.gr-button-primary:active {
    transform: translateY(-4px) scale(1.02) !important;
}

/* 🎭 Emotion Card - Spectacular */
.emotion-card {
    background: linear-gradient(135deg, rgba(255, 110, 199, 0.9), rgba(167, 139, 250, 0.9), rgba(96, 165, 250, 0.9)) !important;
    padding: 3rem;
    border-radius: 30px;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 20px 60px rgba(255, 110, 199, 0.4), 0 0 100px rgba(167, 139, 250, 0.3);
    animation: emotionFloat 3s ease-in-out infinite;
    position: relative;
    overflow: hidden;
    z-index: 1;
}

.emotion-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.2) 10%, transparent 60%);
    animation: orbit 10s linear infinite;
}

@keyframes emotionFloat {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(-10px) rotate(1deg); }
}

@keyframes orbit {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.emotion-title {
    font-size: 4.5rem !important;
    font-weight: 900 !important;
    color: white !important;
    text-shadow: 0 0 30px rgba(255,255,255,0.5), 0 4px 20px rgba(0,0,0,0.3);
    letter-spacing: -2px;
    margin: 0;
    position: relative;
    z-index: 2;
}

.confidence-text {
    font-size: 1.5rem;
    color: rgba(255,255,255,0.95);
    margin-top: 1.5rem;
    font-weight: 600;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    position: relative;
    z-index: 2;
}

/* 🎵 Playlist Carousel - Stunning */
.playlist-carousel {
    display: flex;
    gap: 2rem;
    overflow-x: auto;
    padding: 2rem 1rem;
    scroll-behavior: smooth;
    position: relative;
    z-index: 1;
}

.playlist-carousel::-webkit-scrollbar {
    height: 8px;
}

.playlist-carousel::-webkit-scrollbar-track {
    background: rgba(167, 139, 250, 0.1);
    border-radius: 10px;
}

.playlist-carousel::-webkit-scrollbar-thumb {
    background: linear-gradient(90deg, var(--aurora-pink), var(--aurora-purple));
    border-radius: 10px;
}

/* 💿 Track Cards - 3D Effect */
.track-card {
    flex: 0 0 220px;
    background: var(--dream-card);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    overflow: hidden;
    cursor: pointer;
    transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    border: 2px solid transparent;
    position: relative;
    transform-style: preserve-3d;
    perspective: 1000px;
}

.track-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, rgba(255,110,199,0.2), rgba(96,165,250,0.2));
    opacity: 0;
    transition: opacity 0.4s;
    z-index: 0;
}

.track-card:hover {
    transform: translateY(-15px) scale(1.05) rotateX(5deg);
    border-color: var(--aurora-pink);
    box-shadow: 
        0 20px 60px rgba(255, 110, 199, 0.6),
        0 0 80px rgba(167, 139, 250, 0.4),
        inset 0 -3px 20px rgba(255,110,199,0.3);
}

.track-card:hover::before {
    opacity: 1;
}

.track-card.playing {
    border-color: var(--aurora-pink);
    animation: playing 2s ease-in-out infinite;
}

@keyframes playing {
    0%, 100% {
        box-shadow: 0 0 40px rgba(255, 110, 199, 0.8), 0 0 80px rgba(167, 139, 250, 0.6);
    }
    50% {
        box-shadow: 0 0 60px rgba(255, 110, 199, 1), 0 0 100px rgba(167, 139, 250, 0.8);
    }
}

.track-image {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    display: block;
    transition: transform 0.5s;
}

.track-card:hover .track-image {
    transform: scale(1.1) rotate(2deg);
}

.track-info {
    padding: 1.2rem;
    position: relative;
    z-index: 2;
    background: linear-gradient(to top, rgba(10, 14, 39, 0.95), transparent);
}

.track-name {
    font-weight: 700;
    color: white;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-size: 1rem;
    margin-bottom: 0.4rem;
    text-shadow: 0 2px 10px rgba(0,0,0,0.5);
}

.track-artist {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.8);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* 📊 Score Distribution - Beautiful */
.score-row {
    margin-bottom: 1rem;
    animation: scoreSlide 0.6s ease-out backwards;
}

.score-row:nth-child(2) { animation-delay: 0.1s; }
.score-row:nth-child(3) { animation-delay: 0.2s; }
.score-row:nth-child(4) { animation-delay: 0.3s; }
.score-row:nth-child(5) { animation-delay: 0.4s; }

@keyframes scoreSlide {
    from {
        opacity: 0;
        transform: translateX(-30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

.score-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
    color: white;
    font-weight: 500;
}

.score-fill {
    height: 10px;
    background: linear-gradient(90deg, var(--aurora-pink), var(--aurora-purple), var(--aurora-blue));
    border-radius: 10px;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 20px rgba(255, 110, 199, 0.5);
    position: relative;
    overflow: hidden;
}

.score-fill::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    animation: scoreShine 2s infinite;
}

@keyframes scoreShine {
    0% { left: -100%; }
    100% { left: 200%; }
}

/* 🎯 Hide scrollbars elegantly */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, var(--aurora-pink), var(--aurora-purple));
    border-radius: 10px;
}

/* ⚡ Loading Animation */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(40px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in {
    animation: fadeInUp 0.8s ease-out backwards;
}

/* 💫 Sparkle Effect */
.sparkle {
    position: absolute;
    width: 4px;
    height: 4px;
    background: white;
    border-radius: 50%;
    pointer-events: none;
    animation: sparkle 1.5s ease-in-out infinite;
}

@keyframes sparkle {
    0%, 100% {
        opacity: 0;
        transform: scale(0);
    }
    50% {
        opacity: 1;
        transform: scale(1);
    }
}
"""

DREAMLAND_JS = """
let currentAudio = null;
let currentCardId = null;

// 🎵 Premium Audio Handler
function playPreview(url, cardId) {
    if (currentAudio) {
        currentAudio.pause();
        if (currentCardId) {
            const prevCard = document.getElementById(currentCardId);
            if (prevCard) prevCard.classList.remove('playing');
        }
    }
    
    if (currentCardId === cardId) {
        currentAudio = null;
        currentCardId = null;
        return;
    }
    
    const audio = new Audio(url);
    audio.volume = 0.7;
    
    const card = document.getElementById(cardId);
    
    audio.onplay = () => {
        if (card) card.classList.add('playing');
    };
    
    audio.onended = () => {
        if (card) card.classList.remove('playing');
        currentAudio = null;
        currentCardId = null;
    };
    
    audio.onerror = () => {
        console.error("Audio error");
       if (card) card.classList.remove('playing');
    };
    
    audio.play().catch(err => console.error("Play failed:", err));
    currentAudio = audio;
    currentCardId = cardId;
}

// ✨ Add sparkles on hover
document.addEventListener('DOMContentLoaded', () => {
    document.addEventListener('mousemove', (e) => {
        if (Math.random() > 0.95) {
            const sparkle = document.createElement('div');
            sparkle.className = 'sparkle';
            sparkle.style.left = e.pageX + 'px';
            sparkle.style.top = e.pageY + 'px';
            document.body.appendChild(sparkle);
            setTimeout(() => sparkle.remove(), 1500);
        }
    });
});
"""

def predict_emotion(audio_input, video_input, progress=gr.Progress()):
    """Dreamland prediction with gorgeous feedback and FFmpeg handling."""
    try:
        progress(0.1, desc="✨ Preparing magic...")
        
        if audio_input is None:
            return (
                "<div style='color: #ff6ec7; padding: 2rem; text-align: center; font-size: 1.2rem; animation: fadeInUp 0.5s;'>🎤 Please record or upload audio</div>",
                "<div style='color: #a78bfa;'>Awaiting your input...</div>",
                "Error"
            )
        
        if video_input is None:
            return (
                "<div style='color: #ff6ec7; padding: 2rem; text-align: center; font-size: 1.2rem; animation: fadeInUp 0.5s;'>📹 Please record or upload video</div>",
                "<div style='color: #a78bfa;'>Awaiting your input...</div>",
                "Error"
            )
        
        progress(0.3, desc="🎵 Analyzing audio waves...")
        time.sleep(0.2)
        
        progress(0.5, desc="🎥 Reading facial expressions...")
        time.sleep(0.2)
        
        progress(0.7, desc="🧠 Fusing magical insights...")
        
        # Try prediction with FFmpeg error handling
        try:
            result = predict_from_live(audio_input, video_input)
        except Exception as pred_error:
            error_str = str(pred_error)
            # Check if it's an FFmpeg error
            if 'ffmpeg' in error_str.lower() or 'FFExecutableNotFoundError' in str(type(pred_error)):
                return (
                    """<div style='background: linear-gradient(135deg, rgba(255,110,199,0.2), rgba(167,139,250,0.2)); 
                              padding: 2.5rem; border-radius: 20px; text-align: center; border: 2px solid var(--aurora-pink);'>
                        <h2 style='color: #ff6ec7; margin: 0 0 1rem 0;'>📹 Webcam Requires FFmpeg</h2>
                        <p style='color: white; font-size: 1.1rem; margin-bottom: 1.5rem;'>
                            To use webcam recording, please install FFmpeg:
                        </p>
                        <div style='background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                            <code style='color: #60a5fa; font-size: 1rem;'>winget install ffmpeg</code>
                        </div>
                        <p style='color: rgba(255,255,255,0.8); margin-top: 1.5rem;'>
                            Or download from: <a href='https://ffmpeg.org/download.html' target='_blank' 
                            style='color: var(--aurora-blue); text-decoration: underline;'>ffmpeg.org</a>
                        </p>
                        <p style='color: #10b981; margin-top: 1.5rem; font-weight: 600;'>
                            💡 TIP: Upload a pre-recorded video file to work without FFmpeg!
                        </p>
                    </div>""",
                    "<div style='color: #a78bfa; text-align: center; padding: 2rem;'>Try uploading a video file instead</div>",
                    "FFmpeg Required"
                )
            else:
                # Other prediction error
                raise pred_error
        
        if 'error' in result:
            return (
                f"<div style='color: #ff6ec7; padding: 2rem; text-align: center;'>⚠️ {result['error']}</div>",
                "<div style='color: #a78bfa;'>Please try again</div>",
                "Error"
            )
        
        emotion = result['emotion']
        confidence = result['confidence']
        scores = result['scores']
        
        progress(0.9, desc="🎵 Curating your playlist...")
        tracks = spotify_client.get_recommendations(emotion, limit=5)
        
        # Generate dreamy HTML
        emotion_html = f"""
        <div class="emotion-card fade-in">
            <h1 class="emotion-title">{emotion.upper()}</h1>
            <p class="confidence-text">Feeling {get_emotion_description(emotion).lower()}</p>
            <div class="confidence-text" style="margin-top: 2rem; font-size: 2rem;">
                <strong>{confidence*100:.1f}%</strong> Match
            </div>
        </div>
        """
        
        # Scores
        scores_html = "<div style='background: var(--dream-card); backdrop-filter: blur(20px); padding: 2rem; border-radius: 25px; margin-top: 1.5rem; border: 1px solid rgba(167,139,250,0.2);'>"
        scores_html += "<h3 style='margin: 0 0 1.5rem 0; color: white; font-size: 1.5rem;'>📊 Emotion Spectrum</h3>"
        
        for em, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]:
            width = (score / max(scores.values())) * 100
            scores_html += f"""
            <div class="score-row">
                <div class="score-label">
                    <span>{em.capitalize()}</span>
                    <span style="font-weight: 700;">{score*100:.1f}%</span>
                </div>
                <div style="background: rgba(167,139,250,0.1); border-radius: 10px; height: 10px; overflow: hidden;">
                    <div class="score-fill" style="width: {width}%"></div>
                </div>
            </div>
            """
        scores_html += "</div>"
        
        # Playlist
        playlist_html = "<div class='playlist-carousel fade-in'>"
        for i, track in enumerate(tracks):
            img = track.get('image', 'https://picsum.photos/seed/music/220/220')
            preview = track.get('preview_url')
            spotify_url = track.get('spotify_url', '#')
            
            onclick = f"playPreview('{preview}', 'card-{i}')" if preview else f"window.open('{spotify_url}', '_blank')"
            
            playlist_html += f"""
            <div id="card-{i}" class="track-card" onclick="{onclick}">
                <img src="{img}" class="track-image" loading="lazy" alt="{track.get('name', 'Track')}" />
                <div class="track-info">
                    <div class="track-name">{track.get('name', 'Unknown')}</div>
                    <div class="track-artist">{track.get('artist', 'Unknown Artist')}</div>
                </div>
            </div>
            """
        playlist_html += "</div>"
        
        progress(1.0, desc="✨ Complete!")
        
        return (emotion_html + scores_html, playlist_html, f"✨ {emotion.capitalize()}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            f"<div style='color: #ff6ec7; padding: 2rem; text-align: center;'>💫 Unexpected error: {str(e)}<br><small style='color: rgba(255,255,255,0.6);'>Check console for details</small></div>",
            "<div style='color: #a78bfa;'>Please try again</div>",
            "Error"
        )

# 🌌 Create Dreamland Demo
with gr.Blocks(title="Music Recommendation System") as demo:
    
    gr.HTML(f"<style>{DREAMLAND_CSS}</style>")
    gr.HTML(f"<script>{DREAMLAND_JS}</script>")
    
    gr.HTML("""
    <div style='text-align: center; margin: 3rem 0;'>
        <h1 style='font-size: 4rem; margin: 0; letter-spacing: 2px;'>Music Recommendation System</h1>
        <div style='height: 4px; width: 200px; margin: 1.5rem auto; background: linear-gradient(90deg, var(--aurora-pink), var(--aurora-purple), var(--aurora-blue)); border-radius: 2px; box-shadow: 0 0 20px rgba(167,139,250,0.5);'></div>
        <p style='color: rgba(255,255,255,0.7); font-size: 1.1rem; margin-top: 1rem; font-weight: 300; letter-spacing: 3px;'>
            EMOTION • ANALYSIS • MUSIC
        </p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📹 Video Input")
            video_input = gr.File(
                label="✨ Upload Video File",
                file_types=["video"],
                type="filepath"
            )
            gr.Markdown("*Upload video (3-5 seconds, .mp4 or .webm)*", elem_classes=["fade-in"])
        
        with gr.Column():
            gr.Markdown("### 🎤 Audio Input")
            audio_input = gr.Audio(
                label="🎙️ Record Your Voice",
                sources=["microphone", "upload"],
                type="filepath"
            )
            gr.Markdown("*3-4 seconds of audio*", elem_classes=["fade-in"])
    
    predict_btn = gr.Button("🚀 Discover Your Emotion", variant="primary", size="lg")
    status_output = gr.Textbox(label="Status", interactive=False, visible=False)
    
    gr.HTML("<hr style='border: none; border-top: 1px solid rgba(167,139,250,0.2); margin: 3rem 0;'>")
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 🎭 Your Emotional Journey")
            emotion_output = gr.HTML()
        
        with gr.Column(scale=4):
            gr.Markdown("### 🎵 Your Perfect Soundtrack")
            playlist_output = gr.HTML()
    
    predict_btn.click(
        fn=predict_emotion,
        inputs=[audio_input, video_input],
        outputs=[emotion_output, playlist_output, status_output]
    )
    
    gr.HTML("""
    <div style='text-align: center; color: rgba(255,255,255,0.6); margin-top: 4rem; padding: 2rem; border-top: 1px solid rgba(167,139,250,0.1); position: relative; z-index: 1;'>
        <p style='margin: 0; font-size: 1.1rem;'>✨ Powered by TensorFlow, Gradio & Spotify</p>
        <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>83% Fusion Accuracy | Built with 💜</p>
    </div>
    """)

if __name__ == "__main__":
    print("=" * 70)
    print("🌌 MUSIC RECOMMENDATION SYSTEM 🌌")
    print("=" * 70)
    print("✨ Loading magical models...")
    
    from fusion_inference import load_model_and_labels
    load_model_and_labels()
    
    print("✅ Models ready!")
    print("🌟 Launching on http://localhost:7862...")
    print("💡 Webcam enabled (requires FFmpeg for recording)")
    print("=" * 70)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False
    )
