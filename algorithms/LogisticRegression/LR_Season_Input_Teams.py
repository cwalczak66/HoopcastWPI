import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv('LogisticRegression/nba_games_sorted_by_date.csv')

def predict_winner_for_teams(team, team_opp):
    # Filter the data for the seasons between 2016 and 2022
    season_data = df[(df['season'] >= 2016) & (df['season'] <= 2022)].copy()

    # Ensure 'date' is in datetime format and sort by date
    season_data['date'] = pd.to_datetime(season_data['date'])
    season_data = season_data.sort_values(by='date').reset_index(drop=True)

    # Prepare the training data
    X_train = season_data.drop(columns=['pts', 'won', 'team', 'team_opp', 'total', 'total_opp', 'date', 'season'])
    y_train = season_data['won']  # Use the 'won' column as the target

    # One-hot encode the 'team' and 'team_opp' columns in the training data
    team_data_train = pd.get_dummies(season_data[['team', 'team_opp']])
    X_train = pd.concat([X_train, team_data_train], axis=1)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter
        'solver': ['lbfgs', 'liblinear']  # Different solvers for logistic regression
    }
    model = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
    model.fit(X_train_scaled, y_train)

    # Create a dataframe for the specific game prediction using the two input teams
    game_data = pd.DataFrame([[team, team_opp]], columns=['team', 'team_opp'])

    # One-hot encode the 'team' and 'team_opp' columns in the game data
    game_data_encoded = pd.get_dummies(game_data)

    # Reindex the columns to ensure the game data matches the training set
    game_data_encoded = game_data_encoded.reindex(columns=X_train.columns, fill_value=0)

    # Feature scaling for game data
    game_data_scaled = scaler.transform(game_data_encoded)

    # Predict the outcome for the specific matchup
    y_pred = model.predict(game_data_scaled)

    # Predict the probability of each team winning
    y_prob = model.predict_proba(game_data_scaled)

    # Determine the predicted winner and loser
    predicted_winner = team if y_pred[0] == 1 else team_opp
    predicted_loser = team_opp if y_pred[0] == 1 else team

    # Round the probabilities to two decimal places before returning
    home_team_prob = round(y_prob[0][1] * 100, 2)  # Probability that the home team wins
    away_team_prob = round(y_prob[0][0] * 100, 2)  # Probability that the away team wins

    # Return the predicted winner, loser, and probabilities
    return predicted_winner, predicted_loser, home_team_prob, away_team_prob
