import pandas as pd
import os
input_file = "data/dataset.csv"

if not os.path.exists(input_file):
    print(f"Error: Could not find {input_file}")
    print("Please ensure your downloaded file is renamed to 'spotify_tracks.csv' and placed in the 'data' folder.")
    exit()

print("Loading raw data... this might take a minute.")
df = pd.read_csv(input_file)

df = df.rename(columns={'artists': 'artist', 'track_name': 'song'})

required_columns = ['artist', 'song', 'danceability', 'energy', 'tempo', 'valence', 'loudness', 'track_id', 'track_genre']

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    print(f"Error: The dataset is still missing these columns: {missing_cols}")
    print(f"Available columns in your file: {list(df.columns)}")
    exit()

df_clean = df[required_columns]

df_clean = df_clean.drop_duplicates(subset=['artist', 'song'])
df_small = df_clean.sample(n=10000, random_state=42)

df_small = df_small.reset_index(drop=True)

output_path = "data/songs_sampled.csv"
df_small.to_csv(output_path, index=True, index_label="song_index")

print(f"Success! Saved {len(df_small)} songs to {output_path}")