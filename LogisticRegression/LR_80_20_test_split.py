import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('LogisticRegression/nba_games_sorted_by_date.csv')

# Function to split the data, train the model, and test it
def run_season_regression(season):
    # Filter the data for the given season
    season_data = df[df['season'] == season].copy()

    # Ensure 'date' is in datetime format and sort by date
    season_data['date'] = pd.to_datetime(season_data['date'])
    season_data = season_data.sort_values(by='date')

    # Perform the 80/20 split
    split_index = int(0.8 * len(season_data))

    # Prepare the training data (first 80%)
    train_data = season_data.iloc[:split_index]

    # Use all features except those directly related to the outcome (won, team, team_opp, pts, total, etc.)
    X_train = train_data.drop(columns=['won', 'team', 'team_opp', 'date', 'season', 'pts', 'pts_opp', 'total', 'total_opp'])
    n = []
    # One-hot encode the 'team' and 'team_opp' columns for training
    team_data_train = pd.get_dummies(train_data[['team', 'team_opp']])
    X_train = pd.concat([X_train, team_data_train], axis=1)

    # The target column is 'won'
    y_train = train_data['won']

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Prepare the test data (last 20%)
    test_data = season_data.iloc[split_index:].copy()

    # Keep 'team', 'team_opp', 'date', 'season', and 'won' for the test data
    y_test = test_data['won']
    test_data = test_data[['team', 'team_opp', 'date', 'season', 'won']]

    # One-hot encode the 'team' and 'team_opp' columns in the test data
    X_test = pd.get_dummies(test_data[['team', 'team_opp']])

    # Ensure the test set has the same columns as the training set
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Predict the outcomes for the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # Compare predicted results to actual results
    print("\nResults for the test set:")
    for i in range(len(y_pred)):
        row = test_data.iloc[i]
        predicted_winner = row['team'] if y_pred[i] == 1 else row['team_opp']
        actual_winner = row['team'] if row['won'] == 1 else row['team_opp']

        print(f"Game {i + 1}: {row['team']} vs {row['team_opp']}")
        print(f"Predicted winner: {predicted_winner}, Actual winner: {actual_winner}")
        print('-' * 40)

    # Print overall accuracy
    print(f"\nOverall Accuracy for season {season}: {accuracy * 100:.2f}%")

# Example usage:
season = int(input("Enter the season (year): "))
run_season_regression(season)
