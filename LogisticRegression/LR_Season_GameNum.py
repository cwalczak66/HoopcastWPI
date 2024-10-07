import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('LogisticRegression/nba_games_sorted_by_date.csv')

# Function to train the model on all previous games and predict the winner for a specific game
def predict_winner_for_game(season, game_number):
    # Filter the data for the given season
    season_data = df[df['season'] == season].copy()

    # Ensure 'date' is in datetime format and sorted chronologically (already sorted by date)
    season_data['date'] = pd.to_datetime(season_data['date'])
    season_data = season_data.sort_values(by='date').reset_index(drop=True)

    # Check if game number is valid
    if game_number > len(season_data) or game_number <= 0:
        print(f"Error: Game number {game_number} is out of range for season {season}.")
        return

    # Split the data for training (all games before the given game number) and testing (the game at game_number)
    X_train = season_data.drop(columns=['pts', 'won', 'team', 'team_opp', 'total', 'total_opp', 'date', 'season']).iloc[:game_number - 1]
    y_train = season_data['won'].iloc[:game_number - 1]  # Use the 'won' column as the target

    # Print the exact game numbers being used for training (based on row index)
    start_game = 1
    end_game = game_number - 1
    print(f"Training model on games {start_game} to {end_game} for season {season}.")

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # For the specific game number, use only the team codes ('team', 'team_opp') for testing
    game_to_predict = season_data[['team', 'team_opp', 'won']].iloc[game_number - 1]

    # Prepare test data for the specific game
    X_test = season_data.drop(columns=['pts', 'won', 'team', 'team_opp', 'total', 'total_opp', 'date', 'season']).iloc[[game_number - 1]]
    
    # Predict the outcome for the specific game
    y_pred = model.predict(X_test)

    # Determine the predicted winner
    predicted_winner = game_to_predict['team'] if y_pred[0] == 1 else game_to_predict['team_opp']
    real_winner = game_to_predict['team'] if game_to_predict['won'] == 1 else game_to_predict['team_opp']

    # Print results
    print(f"\nPrediction for Game {game_number} in season {season}:")
    print(f"{game_to_predict['team']} vs {game_to_predict['team_opp']}")
    print(f"Predicted winner: {predicted_winner}")
    print(f"Real winner: {real_winner}")

# Get user input for the season and game number
season = int(input("Enter the season (year): "))
game_number = int(input(f"Enter the game number for season {season}: "))
predict_winner_for_game(season, game_number)
