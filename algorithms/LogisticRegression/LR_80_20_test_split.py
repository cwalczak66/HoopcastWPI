import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the dataset
df = pd.read_csv('LogisticRegression/nba_games_sorted_by_date.csv')

# Function to split the data, train the model, and test it
def run_season_regression(season):
    # Filter the data for the given season
    season_data = df[df['season'] == season].copy()

    # Ensure 'date' is in datetime format and sort by date
    season_data['date'] = pd.to_datetime(season_data['date'])
    season_data = season_data.sort_values(by='date')

    # Use all features except those directly related to the outcome (won, team, team_opp, pts, total, etc.)
    X = season_data.drop(columns=['won', 'team', 'team_opp', 'date', 'season', 'pts', 'pts_opp', 'total', 'total_opp'])
    y = season_data['won']

    # One-hot encode the 'team' and 'team_opp' columns
    team_data = pd.get_dummies(season_data[['team', 'team_opp']])
    X = pd.concat([X, team_data], axis=1)

    # Save the indices before the train-test split for later referencing
    original_indices = season_data.index

    # Perform train-test split and preserve the index
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, original_indices, test_size=0.2, random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the logistic regression model with hyperparameter tuning
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter
        'solver': ['lbfgs', 'liblinear']  # Different solvers for logistic regression
    }
    model = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
    model.fit(X_train_scaled, y_train)

    # Predict the outcomes for the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # Print the best parameters
    # print(f"Best Hyperparameters: {model.best_params_}")

    # # Compare predicted results to actual results
    # print("\nResults for the test set:")
    
    # Use the stored indices to refer back to the original test data
    test_data = season_data.loc[test_indices]
    
    for i in range(len(y_pred)):
        row = test_data.iloc[i]
        predicted_winner = row['team'] if y_pred[i] == 1 else row['team_opp']
        actual_winner = row['team'] if row['won'] == 1 else row['team_opp']

        # print(f"Game {i + 1}: {row['team']} vs {row['team_opp']}")
        # print(f"Predicted winner: {predicted_winner}, Actual winner: {actual_winner}")
        # print('-' * 40)

    # Print overall accuracy
    print(f"\nOverall Accuracy for season {season}: {accuracy * 100:.2f}%")

# Example usage:
for season in range(2016, 2023):
    run_season_regression(season)