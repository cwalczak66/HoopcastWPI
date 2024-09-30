import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset (replace with your actual file path)
data = pd.read_csv('TestData2.csv')

# Create dummy variables for TeamA and TeamB (categorical variables)
teamA_dummies = pd.get_dummies(data['TeamA'], prefix='TeamA', dtype=float)
teamB_dummies = pd.get_dummies(data['TeamB'], prefix='TeamB', dtype=float)

# Combine dummy variables into a feature set
X = pd.concat([teamA_dummies, teamB_dummies], axis=1)

# Ensure the data is cast to the correct numeric types
X = X.astype(float)

# Target variables (Points scored by TeamA and TeamB)
y_A = data['PointsA'].astype(float)
y_B = data['PointsB'].astype(float)

# Train-test split
X_train, X_test, y_A_train, y_A_test = train_test_split(X, y_A, test_size=0.2, random_state=42)
_, _, y_B_train, y_B_test = train_test_split(X, y_B, test_size=0.2, random_state=42)

# Add a constant term (intercept) to the feature set
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Poisson Regression for TeamA Points
poisson_model_A = sm.GLM(y_A_train, X_train, family=sm.families.Poisson()).fit()
print(poisson_model_A.summary())

# Poisson Regression for TeamB Points
poisson_model_B = sm.GLM(y_B_train, X_train, family=sm.families.Poisson()).fit()
print(poisson_model_B.summary())

# Predict on test set for TeamA and TeamB points
y_A_pred = poisson_model_A.predict(X_test)
y_B_pred = poisson_model_B.predict(X_test)

# Evaluate the model using Mean Squared Error
mse_A = mean_squared_error(y_A_test, y_A_pred)
mse_B = mean_squared_error(y_B_test, y_B_pred)

print(f'TeamA prediction MSE: {mse_A}')
print(f'TeamB prediction MSE: {mse_B}')

# Convert predictions to integers (points scored are whole numbers)
y_A_pred_rounded = np.round(y_A_pred)
y_B_pred_rounded = np.round(y_B_pred)

# Show some example predictions
predictions = pd.DataFrame({
    'TeamA Predicted Points': y_A_pred_rounded,
    'TeamB Predicted Points': y_B_pred_rounded,
    'Actual TeamA Points': y_A_test,
    'Actual TeamB Points': y_B_test
})

print(predictions.head())
