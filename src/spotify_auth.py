# src/spotify_auth.py
"""
Spotify Web API integration with robust token management and playback support.
"""
import os
import time
import requests
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyClient:
    def __init__(self):
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        self.redirect_uri = os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:7861/callback')
        self._access_token = None
        self._token_expiry = 0
        
        # Emotion to seed mapping
        self.EMOTION_SEEDS = {
            'calm': {'genres': ['chill', 'ambient', 'lo-fi'], 'valence': 0.3, 'energy': 0.3},
            'happy': {'genres': ['pop', 'funk', 'dance'], 'valence': 0.9, 'energy': 0.8},
            'angry': {'genres': ['rock', 'metal', 'punk'], 'valence': 0.2, 'energy': 0.9},
            'sad': {'genres': ['indie', 'acoustic', 'singer-songwriter'], 'valence': 0.2, 'energy': 0.3},
            'surprise': {'genres': ['electronic', 'experimental', 'indie'], 'valence': 0.6, 'energy': 0.7},
            'neutral': {'genres': ['electronic', 'minimal', 'ambient'], 'valence': 0.5, 'energy': 0.5},
            'fear': {'genres': ['ambient', 'dark-ambient', 'soundtrack'], 'valence': 0.3, 'energy': 0.4},
            'disgust': {'genres': ['alternative', 'grunge', 'industrial'], 'valence': 0.3, 'energy': 0.6}
        }
        
        # Fallback playlists with working images (using Picsum for reliability)
        self.FALLBACK_PLAYLISTS = {
            'calm': [
                {'name': 'Weightless', 'artist': 'Marconi Union', 'preview_url': None, 'image': 'https://picsum.photos/seed/MarconiUnion/200/200', 'spotify_url': 'https://open.spotify.com/track/6YXbjRWPhjUmqe3R8C3YPQ'},
                {'name': 'Clair de Lune', 'artist': 'Claude Debussy', 'preview_url': None, 'image': 'https://picsum.photos/seed/ClaudeDebussy/200/200', 'spotify_url': 'https://open.spotify.com/track/5Z01UMMf7V1o0MzF86s6WJ'},
                {'name': 'Breathe', 'artist': 'Télépopmusik', 'preview_url': None, 'image': 'https://picsum.photos/seed/Telepopmusik/200/200', 'spotify_url': 'https://open.spotify.com/track/1r4WZSnKpr4VgcO1Hqx0Ei'},
                {'name': 'Pure Shores', 'artist': 'All Saints', 'preview_url': None, 'image': 'https://picsum.photos/seed/AllSaints/200/200', 'spotify_url': 'https://open.spotify.com/track/0WO6a4oii5GW5mBNfrxWTi'},
                {'name': 'Sunset Lover', 'artist': 'Petit Biscuit', 'preview_url': None, 'image': 'https://picsum.photos/seed/PetitBiscuit/200/200', 'spotify_url': 'https://open.spotify.com/track/0Z7fMxuAXkq8CjNZfniM0K'}
            ],
            'happy': [
                {'name': 'Happy', 'artist': 'Pharrell Williams', 'preview_url': None, 'image': 'https://picsum.photos/seed/PharrellWilliams/200/200', 'spotify_url': 'https://open.spotify.com/track/60nZcImufyMA1MKQY3dcCH'},
                {'name': 'Good Vibrations', 'artist': 'The Beach Boys', 'preview_url': None, 'image': 'https://picsum.photos/seed/TheBeachBoys/200/200', 'spotify_url': 'https://open.spotify.com/track/54KqEKEzT1hS1fLU2yNhXQ'},
                {'name': 'Walking on Sunshine', 'artist': 'Katrina and the Waves', 'preview_url': None, 'image': 'https://picsum.photos/seed/Katrina/200/200', 'spotify_url': 'https://open.spotify.com/track/05wIrZSwuaVWhcv5FfqeH0'},
                {'name': 'Don\'t Stop Me Now', 'artist': 'Queen', 'preview_url': None, 'image': 'https://picsum.photos/seed/Queen/200/200', 'spotify_url': 'https://open.spotify.com/track/7hQJA50XrCWABAu5v6QZ4i'},
                {'name': 'Uptown Funk', 'artist': 'Mark Ronson ft. Bruno Mars', 'preview_url': None, 'image': 'https://picsum.photos/seed/BrunoMars/200/200', 'spotify_url': 'https://open.spotify.com/track/32OlwWuMpZ6b0aN2RZOeMS'}
            ],
            'sad': [
                {'name': 'Someone Like You', 'artist': 'Adele', 'preview_url': None, 'image': 'https://picsum.photos/seed/Adele/200/200', 'spotify_url': 'https://open.spotify.com/track/1zwMYTA5nlNjZxYrvBB2pV'},
                {'name': 'The Night We Met', 'artist': 'Lord Huron', 'preview_url': None, 'image': 'https://picsum.photos/seed/LordHuron/200/200', 'spotify_url': 'https://open.spotify.com/track/0zPBMXgxlP7s3A4D8lZFXx'},
                {'name': 'Skinny Love', 'artist': 'Bon Iver', 'preview_url': None, 'image': 'https://picsum.photos/seed/BonIver/200/200', 'spotify_url': 'https://open.spotify.com/track/01x7A5NnRjOWkViN57w0P5'},
                {'name': 'Mad World', 'artist': 'Gary Jules', 'preview_url': None, 'image': 'https://picsum.photos/seed/GaryJules/200/200', 'spotify_url': 'https://open.spotify.com/track/3JOVTQ5h8HGFnDdp4VT3MP'},
                {'name': 'Hurt', 'artist': 'Johnny Cash', 'preview_url': None, 'image': 'https://picsum.photos/seed/JohnnyCash/200/200', 'spotify_url': 'https://open.spotify.com/track/4ozJorDhRp5Xx97V8LWK72'}
            ],
            'angry': [
                {'name': 'Break Stuff', 'artist': 'Limp Bizkit', 'preview_url': None, 'image': 'https://picsum.photos/seed/LimpBizkit/200/200', 'spotify_url': 'https://open.spotify.com/track/0SGkfp1Yxp9JqNH3ChB9Ll'},
                {'name': 'Killing in the Name', 'artist': 'Rage Against the Machine', 'preview_url': None, 'image': 'https://picsum.photos/seed/RATM/200/200', 'spotify_url': 'https://open.spotify.com/track/59WN2psbdS5c7LI5gYuqMf'},
                {'name': 'Bodies', 'artist': 'Drowning Pool', 'preview_url': None, 'image': 'https://picsum.photos/seed/DrowningPool/200/200', 'spotify_url': 'https://open.spotify.com/track/1Qz8WG3KcNdhZvxAA8uNOD'},
                {'name': 'Smells Like Teen Spirit', 'artist': 'Nirvana', 'preview_url': None, 'image': 'https://picsum.photos/seed/Nirvana/200/200', 'spotify_url': 'https://open.spotify.com/track/4CeeEOM32jQcH3eN9Q2dGj'},
                {'name': 'Chop Suey!', 'artist': 'System of a Down', 'preview_url': None, 'image': 'https://picsum.photos/seed/SOAD/200/200', 'spotify_url': 'https://open.spotify.com/track/2DlHlPMa4M17kufBvI2lEN'}
            ],
            'neutral': [
                {'name': 'Midnight City', 'artist': 'M83', 'preview_url': None, 'image': 'https://picsum.photos/seed/M83/200/200', 'spotify_url': 'https://open.spotify.com/track/0G21yYKSoU0WQ1HUgg5Rz'},
                {'name': 'Intro', 'artist': 'The xx', 'preview_url': None, 'image': 'https://picsum.photos/seed/Thexx/200/200', 'spotify_url': 'https://open.spotify.com/track/7FIWs0pqAYbP91WWM0vlTQ'},
                {'name': 'Teardrop', 'artist': 'Massive Attack', 'preview_url': None, 'image': 'https://picsum.photos/seed/MassiveAttack/200/200', 'spotify_url': 'https://open.spotify.com/track/7gsWAHLeT0w7es6FofOXk1'},
                {'name': 'Holocene', 'artist': 'Bon Iver', 'preview_url': None, 'image': 'https://picsum.photos/seed/BonIver/200/200', 'spotify_url': 'https://open.spotify.com/track/7pBtnXXlQYhbP3xsVgaK4u'},
                {'name': 'Breathe', 'artist': 'Pink Floyd', 'preview_url': None, 'image': 'https://picsum.photos/seed/PinkFloyd/200/200', 'spotify_url': 'https://open.spotify.com/track/3OBafSVwBB02dsA1N0lDNM'}
            ],
            'surprise': [
                {'name': 'Bohemian Rhapsody', 'artist': 'Queen', 'preview_url': None, 'image': 'https://picsum.photos/seed/Queen/200/200', 'spotify_url': 'https://open.spotify.com/track/4u7EnebtmKWzUH433cf5Qv'},
                {'name': 'Paranoid Android', 'artist': 'Radiohead', 'preview_url': None, 'image': 'https://picsum.photos/seed/Radiohead/200/200', 'spotify_url': 'https://open.spotify.com/track/6LgJvl0Xdtc73RJ1mmpotq'},
                {'name': 'Windowlicker', 'artist': 'Aphex Twin', 'preview_url': None, 'image': 'https://picsum.photos/seed/AphexTwin/200/200', 'spotify_url': 'https://open.spotify.com/track/0E9cbw9T4gL6kDqI0KxUO3'},
                {'name': 'Frontier Psychiatrist', 'artist': 'The Avalanches', 'preview_url': None, 'image': 'https://picsum.photos/seed/TheAvalanches/200/200', 'spotify_url': 'https://open.spotify.com/track/3N8mVmpFFOQgHoZs6IDLNr'},
                {'name': 'Svefn-g-englar', 'artist': 'Sigur Rós', 'preview_url': None, 'image': 'https://picsum.photos/seed/SigurRos/200/200', 'spotify_url': 'https://open.spotify.com/track/7AE3nEKVWF4Y2vjBnusjlb'}
            ],
            'fear': [
                {'name': 'Lux Aeterna', 'artist': 'Clint Mansell', 'preview_url': None, 'image': 'https://picsum.photos/seed/ClintMansell/200/200', 'spotify_url': 'https://open.spotify.com/track/2BqzAODdh06nrFwJQa8eLw'},
                {'name': 'Tubular Bells', 'artist': 'Mike Oldfield', 'preview_url': None, 'image': 'https://picsum.photos/seed/MikeOldfield/200/200', 'spotify_url': 'https://open.spotify.com/track/5IMnY18nW24X2jIVNUtDwh'},
                {'name': 'In the House, In a Heartbeat', 'artist': 'John Murphy', 'preview_url': None, 'image': 'https://picsum.photos/seed/JohnMurphy/200/200', 'spotify_url': 'https://open.spotify.com/track/3WyGq3Z5OuWv06puQqIH3H'},
                {'name': 'Tiptoe Through the Tulips', 'artist': 'Tiny Tim', 'preview_url': None, 'image': 'https://picsum.photos/seed/TinyTim/200/200', 'spotify_url': 'https://open.spotify.com/track/7JfFfX54t0p9XNJg3Z9p8E'},
                {'name': 'Gloomy Sunday', 'artist': 'Billie Holiday', 'preview_url': None, 'image': 'https://picsum.photos/seed/BillieHoliday/200/200', 'spotify_url': 'https://open.spotify.com/track/6nGG33EZaF5wgaWOJoDZ8F'}
            ],
            'disgust': [
                {'name': 'Closer', 'artist': 'Nine Inch Nails', 'preview_url': None, 'image': 'https://picsum.photos/seed/NIN/200/200', 'spotify_url': 'https://open.spotify.com/track/5Xr3e53NGOLMlE6t4Xp1Uv'},
                {'name': 'Head Like a Hole', 'artist': 'Nine Inch Nails', 'preview_url': None, 'image': 'https://picsum.photos/seed/NIN2/200/200', 'spotify_url': 'https://open.spotify.com/track/4RacJRTjjdFB63kIyqdqWA'},
                {'name': 'Du Hast', 'artist': 'Rammstein', 'preview_url': None, 'image': 'https://picsum.photos/seed/Rammstein/200/200', 'spotify_url': 'https://open.spotify.com/track/0RWmXwbHKm2vw2MKwcTzXX'},
                {'name': 'Firestarter', 'artist': 'The Prodigy', 'preview_url': None, 'image': 'https://picsum.photos/seed/TheProdigy/200/200', 'spotify_url': 'https://open.spotify.com/track/0H9rqkMKG7dMeqcP1QJYMm'},
                {'name': 'Black Hole Sun', 'artist': 'Soundgarden', 'preview_url': None, 'image': 'https://picsum.photos/seed/Soundgarden/200/200', 'spotify_url': 'https://open.spotify.com/track/0vFOzaXqZHahrZp6enPrpW'}
            ]
        }

    def get_token(self):
        """Get valid access token, refreshing if necessary."""
        if self._access_token and time.time() < self._token_expiry:
            return self._access_token
            
        if not self.client_id or not self.client_secret:
            logger.warning("Spotify credentials missing")
            return None
            
        try:
            auth_str = f"{self.client_id}:{self.client_secret}"
            b64_auth = base64.b64encode(auth_str.encode()).decode()
            
            response = requests.post(
                'https://accounts.spotify.com/api/token',
                headers={'Authorization': f'Basic {b64_auth}'},
                data={'grant_type': 'client_credentials'},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                self._access_token = data['access_token']
                self._token_expiry = time.time() + data['expires_in'] - 60  # Buffer
                return self._access_token
            else:
                logger.error(f"Token error: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Auth exception: {e}")
            return None

    def get_recommendations(self, emotion, limit=5):
        """Get real Spotify tracks using Search API (works with Client Credentials)."""
        token = self.get_token()
        if not token:
            logger.warning("No token, using fallback")
            return self._get_fallback(emotion, limit)
        
        # Search queries for each emotion (keywords + moods that work well)
        search_queries = {
            'calm': 'chill peaceful relaxing calm meditation spa',
            'happy': 'happy upbeat positive feel good celebration dance',
            'sad': 'sad emotional melancholy heartbreak cry',
            'angry': 'angry aggressive intense rock metal power',
            'neutral': 'indie alternative acoustic mellow',
            'surprise': 'epic dramatic cinematic surprise intense',
            'fear': 'dark suspense thriller horror intense scary',
            'disgust': 'industrial dark aggressive grunge punk'
        }
        
        query = search_queries.get(emotion, 'popular music')
        
        try:
            # Use Search API which definitely works with Client Credentials
            response = requests.get(
                'https://api.spotify.com/v1/search',
                headers={'Authorization': f'Bearer {token}'},
                params={
                    'q': query,
                    'type': 'track',
                    'limit': limit * 2,  # Get more to filter ones with previews
                    'market': 'US'
                },
                timeout=10
            )
            
            logger.info(f"Spotify Search API status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'tracks' not in data or 'items' not in data['tracks']:
                    logger.warning("No tracks in search response")
                    return self._get_fallback(emotion, limit)
                
                all_tracks = []
                tracks_with_preview = []
                
                for t in data['tracks']['items']:
                    track_data = {
                        'id': t.get('id', ''),
                        'name': t.get('name', 'Unknown'),
                        'artist': t['artists'][0]['name'] if t.get('artists') else 'Unknown',
                        'album': t['album']['name'] if t.get('album') else '',
                        'image': t['album']['images'][0]['url'] if t.get('album', {}).get('images') else 'https://picsum.photos/seed/music/220/220',
                        'preview_url': t.get('preview_url'),
                        'spotify_url': t['external_urls']['spotify'] if t.get('external_urls') else '#'
                    }
                    
                    all_tracks.append(track_data)
                    if track_data['preview_url']:  # Prioritize tracks with previews
                        tracks_with_preview.append(track_data)
                
                # Prefer tracks with previews, but include others if needed
                if len(tracks_with_preview) >= limit:
                    result = tracks_with_preview[:limit]
                else:
                    result = tracks_with_preview + all_tracks[:limit - len(tracks_with_preview)]
                
                with_preview = sum(1 for t in result if t['preview_url'])
                logger.info(f"✅ Got {len(result)} REAL Spotify tracks from search ({with_preview}/{len(result)} have preview URLs)")
                return result[:limit]
                
            elif response.status_code == 401:
                logger.error("Spotify auth failed - token expired")
                self.token = None  # Reset token
                return self._get_fallback(emotion, limit)
            else:
                logger.error(f"Spotify Search API error {response.status_code}: {response.text[:200]}")
                return self._get_fallback(emotion, limit)
                
        except Exception as e:
            logger.error(f"Spotify search exception: {type(e).__name__} - {str(e)}")
            return self._get_fallback(emotion, limit)

    def _get_fallback(self, emotion, limit):
        """Return fallback tracks."""
        # Return generic list if specific emotion fallback missing
        return self.FALLBACK_PLAYLISTS.get(emotion, self.FALLBACK_PLAYLISTS.get('happy'))[:limit]

    def proxy_preview(self, url):
        """Stream preview audio bytes."""
        try:
            resp = requests.get(url, stream=True, timeout=10)
            return resp.content, resp.headers.get('content-type')
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            return None, None

# Global instance
spotify_client = SpotifyClient()

def get_recommendations_for_emotion(emotion, limit=5):
    return spotify_client.get_recommendations(emotion, limit)
