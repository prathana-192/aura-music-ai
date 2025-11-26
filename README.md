# 🎵 Aura AI: Transparent Music Intelligence

**Aura AI** is a next-generation music recommendation engine powered by Contextual Multi-Armed Bandits (CMAB). Unlike traditional collaborative filtering systems that require massive user history, Aura AI learns in real-time, balancing Exploration (discovery) and Exploitation (personalization) to solve the "Cold Start" problem.

---

##  The Intelligence Core

At the heart of Aura AI lies the LinUCB (Linear Upper Confidence Bound) algorithm. It treats every recommendation as a decision-making step where the agent estimates the potential reward of a song based on:
1.  **User Preference (theta):** What the user historically likes.
2.  **Uncertainty (alpha):** How much the system doesn't know about a genre.

### Mathematical Formulation
The decision policy is governed by the UCB maximization rule:

$$p_{t,a} = \theta_a^T x_{t,a} + \alpha \sqrt{x_{t,a}^T A_a^{-1} x_{t,a}}$$

Where:
* $x_{t,a}$ is the context vector (Energy, Tempo, Danceability, Mood).
* $A_a$ is the covariance matrix representing confidence.
* $\alpha$ is the hyperparameter controlling curiosity.

---

##  Key Features

* **Zero-Shot Onboarding:** Solves the cold-start problem by pre-training weights on user-selected genres.
* **Glass Box Visualization:** Real-time Radar Charts overlay user preference shapes onto song features, making the AI's logic transparent.
* **Discovery Mode:** A toggleable parameter ($\alpha$) that shifts the AI from "Safe" recommendations to "High Exploration" mode.
* **Contextual Steering:** Users can mask recommendations based on current activity (e.g., "Focus," "Workout").
* **Live Analytics:** Real-time dashboards showing learning curves, preference convergence, and cumulative regret.

---

##  Installation & Setup

### Prerequisites
* Python 3.9 or higher
* Git

### 1. Clone the Repository
```bash
git clone [https://github.com/prathana-192/aura-music-ai.git](https://github.com/prathana-192/aura-music-ai.git)
cd aura-music-ai 

Install Dependencies
pip install -r requirements.txt

Project Structure
aura-music-ai/
├── data/
│   └── songs_sampled.csv    # Curated dataset (9,000+ tracks)
├── .streamlit/
│   └── config.toml          # Dark Mode & UI Theme settings
├── app.py                   # Main Application (Frontend + Logic)
├── recommender.py           # LinUCB Class (The "Brain")
├── comparison.py            # Simulation script for learning curves
├── requirements.txt         # Project dependencies
└── README.md                # Project Documentation

Performance Benchmarks
In simulated environments with 1,000 interaction steps, Aura AI demonstrated:

70% Accuracy in predicting user likes (compared to ~50% for random baselines).

Regret Minimization: Evidence of convergence to optimal policy within 800 steps.

Adaptability: Successfully shifted preference weights when user context changed.

Dataset: Spotify Tracks Dataset (Kaggle), filtered for speechiness and regional relevance.
