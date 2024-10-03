import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Load and prepare the data
data = pd.read_csv('data/TestData2.csv')

# Create dummy variables for TeamA and TeamB
teamA_dummies = pd.get_dummies(data['TeamA'], prefix='TeamA', dtype=float)
teamB_dummies = pd.get_dummies(data['TeamB'], prefix='TeamB', dtype=float)

# Combine dummy variables into a feature set
X = pd.concat([teamA_dummies, teamB_dummies], axis=1)
X = X.astype(float)

# Add constant to X first
X_with_constant = sm.add_constant(X)

# Store the column names including the constant
model_columns = X_with_constant.columns

# Target variables
y_A = data['PointsA'].astype(float)
y_B = data['PointsB'].astype(float)

# Train the models on full dataset
poisson_model_A = sm.GLM(y_A, X_with_constant, family=sm.families.Poisson()).fit()
poisson_model_B = sm.GLM(y_B, X_with_constant, family=sm.families.Poisson()).fit()

def poisson(home_team, away_team):
    """
    Predict the score of a game between two teams.
    
    Args:
        home_team (str): Name of the home team
        away_team (str): Name of the away team
    
    Returns:
        tuple: Predicted scores (home_score, away_score)
    """
    # Create a DataFrame with zeros using the exact same columns as the training data
    # Initialize with zeros including the constant column
    prediction_data = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Set the constant term to 1
    prediction_data['const'] = 1
    
    # Set the appropriate teams to 1
    prediction_data[f'TeamA_{home_team}'] = 1
    prediction_data[f'TeamB_{away_team}'] = 1
    
    # Make predictions
    home_score = poisson_model_A.predict(prediction_data)[0]
    away_score = poisson_model_B.predict(prediction_data)[0]
    
    return round(home_score), round(away_score)