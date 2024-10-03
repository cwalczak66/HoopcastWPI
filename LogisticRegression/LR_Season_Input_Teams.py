import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv('LogisticRegression/nba_games_sorted_by_date.csv')

def predict_winner_for_teams(season, team, team_opp):
    # Filter the data for the given season
    season_data = df[df['season'] == season].copy()

    season_data['date'] = pd.to_datetime(season_data['date'])
    season_data = season_data.sort_values(by='date').reset_index(drop=True)

    X_train = season_data.drop(columns=['pts', 'won', 'team', 'team_opp', 'total', 'total_opp', 'date', 'season'])
    y_train = season_data['won']  # Use the 'won' column as the target

    # One-hot encode the 'team' and 'team_opp' columns in the training data
    team_data_train = pd.get_dummies(season_data[['team', 'team_opp']])
    X_train = pd.concat([X_train, team_data_train], axis=1)

    # Train the logistic regression model on the entire season
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Create a dataframe for the specific game prediction using the two input teams
    game_data = pd.DataFrame([[team, team_opp]], columns=['team', 'team_opp'])

    # One-hot encode the 'team' and 'team_opp' columns in the game data
    game_data_encoded = pd.get_dummies(game_data)

    # Reindex the columns to ensure the game data matches the training set
    game_data_encoded = game_data_encoded.reindex(columns=X_train.columns, fill_value=0)

    # Predict the outcome for the specific matchup
    y_pred = model.predict(game_data_encoded)

    # Predict the probability of each team winning
    y_prob = model.predict_proba(game_data_encoded)

    # Determine the predicted winner
    predicted_winner = team if y_pred[0] == 1 else team_opp

    # Calculate the probabilities for each team
    home_team_prob = y_prob[0][1] * 100  # Probability that the home team wins
    away_team_prob = y_prob[0][0] * 100  # Probability that the away team wins

    # Print results
    print(f"\nPrediction for {team} (home) vs {team_opp} (away) in season {season}:")
    print(f"Predicted winner: {predicted_winner}")
    print(f"{team} win probability: {home_team_prob:.2f}%")
    print(f"{team_opp} win probability: {away_team_prob:.2f}%")

# Get user input for the season and teams
season = int(input("Enter the season (year): "))
team = input("Enter the home team code (team): ").upper()
team_opp = input("Enter the away team code (team_opp): ").upper()

predict_winner_for_teams(season, team, team_opp)
