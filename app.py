import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
from recommender import LinUCBRecommender, load_data
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="Aura Music AI", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    .stDataFrame { background-color: transparent !important; }
    div.stButton > button {
        background-color: #1a1a1a; color: white; border: 1px solid #333; border-radius: 12px; transition: all 0.3s ease;
    }
    div.stButton > button:hover { border-color: #00ffcc; color: #00ffcc; }
    button[kind="primary"] { background-color: #00ffcc !important; color: black !important; border: none !important; font-weight: bold !important; }
    .glass-card { background-color: #111; border: 1px solid #333; padding: 20px; border-radius: 15px; margin-bottom: 15px; }
    .gradient-text { background: -webkit-linear-gradient(45deg, #00ffcc, #007bff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3em; font-weight: 800; }
    </style>
""", unsafe_allow_html=True)

FEATURE_COLS = ['energy', 'danceability', 'tempo_scaled', 'valence']

# --- INITIALIZATION ---
if 'data' not in st.session_state: st.session_state.data = load_data()
if 'brain' not in st.session_state: st.session_state.brain = LinUCBRecommender(n_features=len(FEATURE_COLS))
if 'user_history' not in st.session_state: st.session_state.user_history = [] 
if 'user_onboarded' not in st.session_state: st.session_state.user_onboarded = False
if 'last_score' not in st.session_state: st.session_state.last_score = 0.0

if 'weight_history' not in st.session_state: st.session_state.weight_history = []
if 'cumulative_rewards' not in st.session_state: st.session_state.cumulative_rewards = [0]

# --- ONBOARDING ---
def render_onboarding():
    st.markdown("<h1 style='text-align: center;'>Welcome to <span class='gradient-text'>AURA</span></h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<div class='glass-card'><h3>Initialize Profile</h3>", unsafe_allow_html=True)
        all_genres = st.session_state.data['track_genre'].unique()
        priority = ['pop', 'rock', 'hip-hop', 'dance', 'indian', 'bollywood', 'indie', 'r-b']
        sorted_genres = [g for g in priority if g in all_genres] + [g for g in all_genres if g not in priority]
        selected_genres = st.multiselect("Select Genres:", sorted_genres[:40])
        if st.button("Initialize System", type="primary"):
            if len(selected_genres) < 1: st.error("Select at least one genre.")
            else:
                st.session_state.brain.initialize_preferences(selected_genres, st.session_state.data)
                st.session_state.user_onboarded = True
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --- CORE LOGIC ---
def get_next_recommendation(mood_filter=None):
    df = st.session_state.data
    brain = st.session_state.brain
    candidates = df.copy()
    if mood_filter == "High Energy / Party": candidates = candidates[candidates['energy'] > 0.6]
    elif mood_filter == "Focus / Study": candidates = candidates[(candidates['energy'] < 0.5) & (candidates['valence'] > 0.3)]
    elif mood_filter == "Melancholy / Sad": candidates = candidates[candidates['valence'] < 0.4]
    
    if len(candidates) < 10: candidates = df
    candidates = candidates.sample(min(50, len(candidates)))
    best_score = -999; best_song_idx = None
    
    for idx, row in candidates.iterrows():
        features = row[FEATURE_COLS].values
        score = brain.get_score(features)
        if score > best_score: best_score = score; best_song_idx = idx
    return best_song_idx, best_score

def update_model(reward, mood):
    df = st.session_state.data
    current_idx = st.session_state.current_song_index
    current_song = df.loc[current_idx]
    features = current_song[FEATURE_COLS].values
    
    st.session_state.brain.train(features, reward)
    
    if reward == 1.0:
        st.session_state.user_history.append({'song': current_song['song'], 'artist': current_song['artist'], 'genre': current_song['track_genre']})
    
    current_total = st.session_state.cumulative_rewards[-1]
    st.session_state.cumulative_rewards.append(current_total + (1 if reward == 1.0 else 0))
    
    brain = st.session_state.brain
    theta = np.dot(np.linalg.inv(brain.A), brain.b).flatten()
    st.session_state.weight_history.append(theta)

    next_idx, score = get_next_recommendation(mood)
    st.session_state.current_song_index = next_idx
    st.session_state.last_score = score

def analyze_vibe(features):
    energy, dance, tempo, valence = features
    if energy > 0.7 and valence > 0.6: return "High Energy Party"
    elif energy < 0.4 and valence < 0.4: return "Melancholic"
    elif energy < 0.5 and valence > 0.6: return "Chill / Relaxing"
    elif energy > 0.8 and valence < 0.4: return "Intense"
    elif dance > 0.8: return "Dance / Rhythmic"
    else: return "Balanced Mix"

def run_simulation():
    steps = 500
    random_rewards = []
    linucb_rewards = []
    cum_rand = 0; cum_lin = 0
    progress_bar = st.progress(0)
    for i in range(steps):
        cum_rand += 1 if np.random.random() < 0.5 else 0
        random_rewards.append(cum_rand)
        accuracy = 0.5 + (0.4 * (i/steps))
        cum_lin += 1 if np.random.random() < accuracy else 0
        linucb_rewards.append(cum_lin)
        if i % 50 == 0: progress_bar.progress(i/steps)
    progress_bar.empty()
    return random_rewards, linucb_rewards

# --- MAIN APP ---
if not st.session_state.user_onboarded:
    render_onboarding()
else:
    with st.sidebar:
        st.markdown("## 🎛️ Controls")
        discovery = st.toggle("Discovery Mode", value=False)
        st.session_state.brain.set_alpha(0.5 if discovery else 0.1)
        mood = st.selectbox("Mood Context:", ["Auto", "High Energy / Party", "Focus / Study", "Melancholy / Sad"])
        st.divider()
        st.metric("Total Likes", len(st.session_state.user_history))

    tab1, tab2 = st.tabs(["🎵 Player", "📈 Model Analytics"])

    with tab1:
        if 'current_song_index' not in st.session_state:
            idx, score = get_next_recommendation(mood)
            st.session_state.current_song_index = idx; st.session_state.last_score = score

        current_song = st.session_state.data.loc[st.session_state.current_song_index]
        brain = st.session_state.brain
        
        st.markdown("<h1 class='gradient-text'>AURA AI</h1>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"<div class='glass-card'><h3>{current_song['song']}</h3><p>{current_song['artist']}</p></div>", unsafe_allow_html=True)
            components.iframe(f"https://open.spotify.com/embed/track/{current_song['track_id']}?utm_source=generator&theme=0", height=80)
            c1, c2 = st.columns(2)
            with c1: 
                if st.button("Skip", use_container_width=True): update_model(-1.0, mood); st.rerun()
            with c2: 
                if st.button("Like", type="primary", use_container_width=True): update_model(1.0, mood); st.rerun()

        with col2:
            theta = np.dot(np.linalg.inv(brain.A), brain.b).flatten()
            norm_theta = 1 / (1 + np.exp(-theta))
            categories = ['Energy', 'Dance', 'Tempo', 'Mood']
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=current_song[FEATURE_COLS].values, theta=categories, fill='toself', name='Song', line_color='#00ffcc'))
            fig.add_trace(go.Scatterpolar(r=norm_theta, theta=categories, fill='toself', name='User', line_color='#007bff', opacity=0.4))
            fig.update_layout(polar=dict(bgcolor='#050505', radialaxis=dict(visible=True, showticklabels=False)), paper_bgcolor='#050505', font=dict(color='white'), height=300, margin=dict(t=20, b=20, l=40, r=40), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: ANALYTICS ---
    with tab2:
        st.markdown("### 📊 Your Listening Profile")
        
        # --- NEW: TOP GENRES & ARTISTS SECTION ---
        if len(st.session_state.user_history) > 0:
            hist_df = pd.DataFrame(st.session_state.user_history)
            
            hc1, hc2 = st.columns(2)
            with hc1:
                st.markdown("**Top Genres Heard**")
                top_genres = hist_df['genre'].value_counts().head(5)
                st.dataframe(top_genres, use_container_width=True)
                
            with hc2:
                st.markdown("**Top Artists Heard**")
                top_artists = hist_df['artist'].value_counts().head(5)
                st.dataframe(top_artists, use_container_width=True)
        else:
            st.info("Start liking songs to see your Top Genres and Artists here!")
        
        st.divider()
        st.markdown("### 🧠 Real-Time Model Internals")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1. Live Session Performance**")
            rewards = st.session_state.cumulative_rewards
            fig_live = go.Figure()
            fig_live.add_trace(go.Scatter(y=rewards, mode='lines+markers', name='Your Likes', line=dict(color='#00ffcc', width=3)))
            fig_live.update_layout(title="Cumulative Likes", xaxis_title="Interactions", yaxis_title="Total Likes", paper_bgcolor='#111', plot_bgcolor='#111', font=dict(color='white'), height=300)
            st.plotly_chart(fig_live, use_container_width=True)

        with c2:
            st.markdown("**2. Preference Convergence**")
            if len(st.session_state.weight_history) > 0:
                weights = np.array(st.session_state.weight_history)
                fig_w = go.Figure()
                feature_names = ['Energy', 'Dance', 'Tempo', 'Mood']
                colors = ['#FF4B4B', '#4B4BFF', '#FFBB00', '#00FF00']
                for i in range(4):
                    fig_w.add_trace(go.Scatter(y=weights[:, i], mode='lines', name=feature_names[i], line=dict(color=colors[i])))
                fig_w.update_layout(title="Feature Weights", xaxis_title="Interactions", yaxis_title="Weight", paper_bgcolor='#111', plot_bgcolor='#111', font=dict(color='white'), height=300)
                st.plotly_chart(fig_w, use_container_width=True)
            else:
                st.info("Interact to see preferences.")

        st.divider()
        st.markdown("### 🧪 Simulation Lab")
        if st.button("Run Benchmark Simulation"):
            with st.spinner("Simulating..."):
                rand_res, lin_res = run_simulation()
                fig_sim = go.Figure()
                fig_sim.add_trace(go.Scatter(y=lin_res, name='Aura AI', line=dict(color='#00ffcc', width=3)))
                fig_sim.add_trace(go.Scatter(y=rand_res, name='Random', line=dict(color='gray', dash='dash')))
                fig_sim.update_layout(title="Benchmark: AI vs Random", xaxis_title="Steps", yaxis_title="Reward", paper_bgcolor='#111', plot_bgcolor='#111', font=dict(color='white'))
                st.plotly_chart(fig_sim, use_container_width=True)