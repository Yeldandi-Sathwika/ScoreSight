from flask import Flask, render_template, request
from predict import predict_match
from player_predict import predict_player

import pandas as pd
import os

# ==========================================
# Flask App
# ==========================================

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

# ==========================================
# Paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ==========================================
# Load Team Data
# ==========================================

team_stats = pd.read_csv(
    os.path.join(MODEL_DIR, "team_stats.csv"),
    index_col=0
)

teams = sorted(team_stats.index.tolist())

# ==========================================
# Load Player Data
# ==========================================

player_stats = pd.read_csv(
    os.path.join(MODEL_DIR, "player_stats.csv")
)

players = sorted(
    player_stats["Name"].unique().tolist()
)

# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")

# ==========================================
# Match Prediction
# ==========================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        home_team = request.form["home_team"]
        away_team = request.form["away_team"]

        # Prevent same team selection
        if home_team == away_team:

            return render_template(
                "match.html",
                teams=teams,
                error="Home Team and Away Team cannot be the same."
            )

        try:

            home_score, away_score = predict_match(
                home_team,
                away_team
            )

            if home_score > away_score:

                winner = home_team

            elif away_score > home_score:

                winner = away_team

            else:

                winner = "Draw"

            return render_template(
                "result.html",
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                winner=winner
            )

        except Exception as e:

            return render_template(
                "match.html",
                teams=teams,
                error=str(e)
            )

    return render_template(
        "match.html",
        teams=teams
    )

# ==========================================
# Player Prediction
# ==========================================

@app.route("/player", methods=["GET", "POST"])
def player():

    if request.method == "POST":

        player_name = request.form["player_name"]

        try:

            result = predict_player(player_name)

            return render_template(
                "player_result.html",
                player=result
            )

        except Exception as e:

            return render_template(
                "player.html",
                players=players,
                error=str(e)
            )

    return render_template(
        "player.html",
        players=players
    )

# ==========================================
# About Page
# ==========================================

@app.route("/about")
def about():

    return render_template("index.html")

# ==========================================
# Run App
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )