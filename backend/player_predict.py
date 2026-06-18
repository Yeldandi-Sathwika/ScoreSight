import os
import joblib
import pandas as pd

# ==========================================
# Paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ==========================================
# Load Model
# ==========================================

model = joblib.load(
    os.path.join(MODEL_DIR, "player_prediction_model.pkl")
)

feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "player_feature_columns.pkl")
)

player_stats = pd.read_csv(
    os.path.join(MODEL_DIR, "player_stats.csv")
)

# ==========================================
# Player Prediction
# ==========================================

def predict_player(player_name):

    # Find player
    player = player_stats[
        player_stats["Name"] == player_name
    ]

    if player.empty:
        raise ValueError("Player not found.")

    player = player.iloc[0]

    # Build feature dataframe
    X = pd.DataFrame([{

        "Club": player["Club"],
        "Position": player["Position"],
        "Age": player["Age"],
        "Appearances": player["Appearances"],
        "Wins": player["Wins"],
        "Losses": player["Losses"],
        "Shots": player["Shots"],
        "Shots on target": player["Shots on target"],
        "Big chances missed": player["Big chances missed"],
        "Tackles": player["Tackles"],
        "Interceptions": player["Interceptions"],
        "Clearances": player["Clearances"],
        "Recoveries": player["Recoveries"],
        "Duels won": player["Duels won"],
        "Duels lost": player["Duels lost"],
        "Passes": player["Passes"],
        "Big chances created": player["Big chances created"],
        "Crosses": player["Crosses"],
        "Yellow cards": player["Yellow cards"],
        "Red cards": player["Red cards"],
        "Fouls": player["Fouls"],
        "Offsides": player["Offsides"]

    }])

    # Arrange columns
    X = X[feature_columns]

    # Prediction
    prediction = model.predict(X)

    predicted_goals = max(
        0,
        int(round(prediction[0][0]))
    )

    predicted_assists = max(
        0,
        int(round(prediction[0][1]))
    )

    # Club & Position display
    club_value = player["Club"]
    position_value = player["Position"]

    if isinstance(club_value, (int, float)):
        club_display = f"Club ID {int(club_value)}"
    else:
        club_display = str(club_value)

    if isinstance(position_value, (int, float)):
        position_display = f"Position ID {int(position_value)}"
    else:
        position_display = str(position_value)

    return {

        "player_name": player["Name"],

        "club": club_display,

        "position": position_display,

        "age": int(player["Age"]),

        "predicted_goals": predicted_goals,

        "predicted_assists": predicted_assists

    }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    sample_player = player_stats.iloc[0]["Name"]

    result = predict_player(sample_player)

    print(result)