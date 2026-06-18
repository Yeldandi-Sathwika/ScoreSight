import os
import joblib
import pandas as pd
import numpy as np

# ==========================
# Paths
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ==========================
# Load Model
# ==========================

model = joblib.load(os.path.join(MODEL_DIR, "match_prediction_model.pkl"))

home_encoder = joblib.load(os.path.join(MODEL_DIR, "home_encoder.pkl"))
away_encoder = joblib.load(os.path.join(MODEL_DIR, "away_encoder.pkl"))
season_encoder = joblib.load(os.path.join(MODEL_DIR, "season_encoder.pkl"))

# ==========================
# Load Team Stats
# ==========================

team_stats = pd.read_csv(
    os.path.join(MODEL_DIR, "team_stats.csv"),
    index_col=0
)

# ==========================
# Prediction Function
# ==========================

def predict_match(home_team, away_team):

    # Check if teams exist
    if home_team not in team_stats.index:
        raise ValueError(f"{home_team} not found in team_stats.csv")

    if away_team not in team_stats.index:
        raise ValueError(f"{away_team} not found in team_stats.csv")

    # Encode values
    try:
        home = home_encoder.transform([home_team])[0]
        away = away_encoder.transform([away_team])[0]
        season = season_encoder.transform(["2015/2016"])[0]
    except Exception as e:
        raise ValueError(f"Encoder Error: {e}")

    stage = 1

    home_avg_goals = float(
        team_stats.loc[home_team, "rolling_home_avg_goals"]
    )

    away_avg_goals = float(
        team_stats.loc[away_team, "rolling_away_avg_goals"]
    )

    home_avg_conceded = float(
        team_stats.loc[home_team, "rolling_home_avg_conceded"]
    )

    away_avg_conceded = float(
        team_stats.loc[away_team, "rolling_away_avg_conceded"]
    )

    X = pd.DataFrame([{
        "season": season,
        "stage": stage,
        "home_team": home,
        "away_team": away,
        "rolling_home_avg_goals": home_avg_goals,
        "rolling_away_avg_goals": away_avg_goals,
        "rolling_home_avg_conceded": home_avg_conceded,
        "rolling_away_avg_conceded": away_avg_conceded
    }])

    prediction = model.predict(X)

    home_score = max(0, int(round(prediction[0][0])))
    away_score = max(0, int(round(prediction[0][1])))

    return home_score, away_score