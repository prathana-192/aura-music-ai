import numpy as np
import pandas as pd

class LinUCBRecommender:
    def __init__(self, n_features=4, alpha=0.1):
        self.alpha = alpha
        self.n_features = n_features
        
        # FIX 1: Force dtype=float
        self.A = np.identity(n_features, dtype=float)
        self.b = np.zeros((n_features, 1), dtype=float)

    def set_alpha(self, new_alpha):
        self.alpha = new_alpha

    def get_score(self, song_features):
        x = np.array(song_features, dtype=float).reshape(-1, 1)
        A_inv = np.linalg.inv(self.A)
        theta = np.dot(A_inv, self.b)
        prediction = np.dot(theta.T, x)[0, 0]
        uncertainty = np.sqrt(np.dot(x.T, np.dot(A_inv, x)))[0, 0]
        return prediction + (self.alpha * uncertainty)

    def train(self, song_features, reward):
        x = np.array(song_features, dtype=float).reshape(-1, 1)
        self.A += np.dot(x, x.T)
        self.b += reward * x

    def initialize_preferences(self, selected_genres, df):
        """Onboarding: Pre-trains the model on selected genres."""
        print(f"Initializing profile with: {selected_genres}")
        subset = df[df['track_genre'].isin(selected_genres)]
        if not subset.empty:
            # Calculate average features of selected genres
            avg_features = subset[['energy', 'danceability', 'tempo_scaled', 'valence']].mean().values
            # Train 5 times to create a strong initial bias
            for _ in range(5):
                self.train(avg_features, reward=1.0)

def load_data():
    try:
        df = pd.read_csv("data/songs_sampled.csv")
    except FileNotFoundError:
        print("Using raw dataset fallback...")
        df = pd.read_csv("data/dataset.csv")
    
    # 1. Rename columns
    if 'track_name' in df.columns:
        df = df.rename(columns={'track_name': 'song', 'artists': 'artist'})
    if 'song' not in df.columns:
        df['song'] = "Unknown Song"

    # --- RESTORED FILTERING LOGIC ---
    
    # 2. LANGUAGE/GENRE FILTER
    # We explicitly keep only these genres to ensure English/Hindi content
    target_genres = [
        'pop', 'rock', 'hip-hop', 'dance', 'edm', 'house', # Western Pop
        'indie', 'alternative', 'acoustic', 'r-b', 'soul', 'country', # Western Other
        'indian', 'bollywood', 'punjabi', 'desi', 'hindustani', 'mandopop' # Indian/Asian
    ]
    
    if 'track_genre' in df.columns:
        # Only keep songs that match our target genres
        df = df[df['track_genre'].isin(target_genres)]
        
    # 3. QUALITY FILTER (Remove Podcasts)
    if 'speechiness' in df.columns:
        df = df[df['speechiness'] < 0.66]

    # 4. Force numeric columns & Handle Missing Data
    numeric_cols = ['energy', 'danceability', 'tempo', 'valence', 'loudness']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # 5. Scale Features
    df['tempo_scaled'] = df['tempo'] / 200.0
    df['loudness_scaled'] = (df['loudness'] + 60) / 60.0
    
    # 6. Reset Index (Crucial after filtering!)
    df = df.reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} songs after filtering.")
    return df