# src/ui_components.py
"""
Reusable HTML components for the Gradio UI.
"""

def get_custom_css():
    """Load custom CSS."""
    try:
        with open('assets/css/custom_ui.css', 'r') as f:
            return f.read()
    except:
        return ""

def get_custom_js():
    """Load custom JavaScript."""
    try:
        with open('assets/js/ui_interactions.js', 'r') as f:
            return f.read()
    except:
        return ""

def create_rain_canvas():
    return '<canvas id="rain-canvas"></canvas>'

def create_emotion_card(emotion, confidence, description, color):
    """Create premium emotion result card."""
    return f"""
    <div class="emotion-card scale-in" style="border-left: 4px solid {color};">
        <div class="emotion-header">
            <h1 class="emotion-title" style="color: {color};">{emotion.upper()}</h1>
            <div class="confidence-badge" style="background: {color}22; color: {color};">
                {confidence*100:.1f}% Confidence
            </div>
        </div>
        <p class="emotion-desc">{description}</p>
        
        <div class="confidence-gauge">
            <div class="confidence-fill" style="width: {confidence*100}%; background: {color};"></div>
        </div>
    </div>
    """

def create_playlist_card(tracks):
    """Create premium Spotify playlist carousel."""
    if not tracks:
        return "<p>No tracks available</p>"
    
    cards_html = ""
    for i, track in enumerate(tracks):
        image_url = track.get('image') or 'https://via.placeholder.com/220x220?text=No+Image'
        preview_url = track.get('preview_url')
        spotify_url = track.get('spotify_url', '#')
        
        # Play button logic
        play_action = ""
        if preview_url:
            play_action = f"onclick=\"playPreview('{preview_url}', 'card-{i}')\""
        else:
            play_action = f"onclick=\"window.open('{spotify_url}', '_blank')\""
            
        cards_html += f"""
        <div id="card-{i}" class="track-card fade-in" role="button" tabindex="0" {play_action}>
            <div class="card-image-wrapper">
                <img src="{image_url}" alt="{track['name']}" loading="lazy" class="track-image" />
                <div class="play-overlay">
                    <span class="play-icon">▶</span>
                </div>
                <div class="playing-indicator">
                    <div class="bar"></div><div class="bar"></div><div class="bar"></div>
                </div>
            </div>
            <div class="track-info">
                <div class="track-name" title="{track['name']}">{track['name']}</div>
                <div class="track-artist" title="{track['artist']}">{track['artist']}</div>
            </div>
        </div>
        """
    
    return f"""
    <div class="playlist-container">
        <div class="playlist-carousel">
            {cards_html}
        </div>
    </div>
    """

def create_scores_chart(scores):
    """Create emotion scores distribution chart."""
    if not scores:
        return ""
    
    bars_html = ""
    max_score = max(scores.values()) if scores else 1.0
    
    for emotion, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        width = (score / max_score) * 100 if max_score > 0 else 0
        bars_html += f"""
        <div class="score-row">
            <div class="score-label">
                <span>{emotion.capitalize()}</span>
                <span class="score-val">{score*100:.1f}%</span>
            </div>
            <div class="score-track">
                <div class="score-fill" style="width: {width}%;"></div>
            </div>
        </div>
        """
    
    return f"""
    <div class="emotion-card fade-in" style="margin-top: 1rem;">
        <h3 style="margin-bottom: 1rem; font-size: 1.1rem;">Emotion Distribution</h3>
        {bars_html}
    </div>
    """
