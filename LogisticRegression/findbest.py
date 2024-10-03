import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

# Load the CSV file into a DataFrame
df = pd.read_csv('nba_games.csv')

# Print the shape of the dataset
print(f"Original data shape: {df.shape}")

# Drop non-numeric or irrelevant columns such as '+/-' columns, but keep 'team', 'team_opp', and 'date'
df = df.drop(columns=['+/-', 'mp_max', 'mp_max.1', '+/-_opp', 'mp_max_opp', 'mp_max_opp.1'])

# Drop rows with missing target values ('pts')
df = df.dropna(subset=['pts'])
print(f"Data shape after dropping rows with missing target values: {df.shape}")

# Define features (X) and target (y), excluding 'team', 'team_opp', and 'date' from feature selection
X = df.drop(columns=['pts', 'season', 'team', 'team_opp', 'date'])  # Features excluding 'team', 'team_opp', and 'date'
y = df['pts']  # Target (Points scored)

# Ensure all remaining columns are numerical
X = X.apply(pd.to_numeric, errors='coerce')

# Drop any rows with missing values resulting from conversion
X = X.dropna()

# Check if features (X) and target (y) have valid data
print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# Feature selection: select the 30 best features using SelectKBest with f_regression
best_features_selector = SelectKBest(score_func=f_regression, k=30)
X_best = best_features_selector.fit_transform(X, y)

# Get the indices of the selected features
selected_feature_indices = best_features_selector.get_support(indices=True)

# Get the column names of the selected features
selected_columns = X.columns[selected_feature_indices]

# Print the 30 best features
print("The 30 best features are:")
print(selected_columns)

# Create a DataFrame with the selected features and add 'team', 'team_opp', 'pts', 'season', and 'date' back
df_best_features = pd.DataFrame(X_best, columns=selected_columns)
df_best_features['team'] = df['team'].values  # Keep the original team codes
df_best_features['team_opp'] = df['team_opp'].values  # Keep the original team_opp codes
df_best_features['pts'] = y.values  # Add the target column 'pts'
df_best_features['season'] = df['season'].values  # Add 'season'
df_best_features['date'] = df['date'].values  # Add 'date'

# Sort by 'season'
df_best_features = df_best_features.sort_values(by='season')

# Save the reduced dataset with the best features, 'team', 'team_opp', 'season', and 'date'
output_file_path = 'nba_games_best_features.csv'
df_best_features.to_csv(output_file_path, index=False)

print(f"CSV saved to {output_file_path}")
