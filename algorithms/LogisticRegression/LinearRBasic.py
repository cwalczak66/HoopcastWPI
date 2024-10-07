import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the CSV file into a DataFrame
df = pd.read_csv('NBA23-24.csv')

# Preprocess the data
# Extract relevant features and target variables
visitor_categories = df['Visitor/Neutral'].astype('category').cat.categories
home_categories = df['Home/Neutral'].astype('category').cat.categories

df['Visitor/Neutral'] = df['Visitor/Neutral'].astype('category').cat.codes
df['Home/Neutral'] = df['Home/Neutral'].astype('category').cat.codes

X = df[['Visitor/Neutral', 'Home/Neutral']]
y_visitor = df['VPTS']
y_home = df['HPTS']

# Perform an 80/20 split with the last 20% being the test set
split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]
y_train_visitor = y_visitor[:split_index]
y_test_visitor = y_visitor[split_index:]
y_train_home = y_home[:split_index]
y_test_home = y_home[split_index:]

# Train the linear regression models
model_visitor = LinearRegression()
model_home = LinearRegression()

model_visitor.fit(X_train, y_train_visitor)
model_home.fit(X_train, y_train_home)

# Function to predict the score and winner
def predict_winner(visitor_team, home_team):
    visitor_code = visitor_categories.get_loc(visitor_team)
    home_code = home_categories.get_loc(home_team)
    
    # Create a DataFrame for prediction to match training format
    prediction_input = pd.DataFrame([[visitor_code, home_code]], columns=['Visitor/Neutral', 'Home/Neutral'])
    
    visitor_score = model_visitor.predict(prediction_input)[0]
    home_score = model_home.predict(prediction_input)[0]
    
    if visitor_score > home_score:
        winner = visitor_team
    else:
        winner = home_team
    
    return visitor_score, home_score, winner

# Initialize counters for accuracy calculation
correct_predictions = 0
total_predictions = len(X_test)

# Test the model on the last 20% (test set)
for i in range(len(X_test)):
    visitor_team = visitor_categories[X_test.iloc[i]['Visitor/Neutral']]
    home_team = home_categories[X_test.iloc[i]['Home/Neutral']]
    
    # Get predicted scores and winner
    visitor_score, home_score, predicted_winner = predict_winner(visitor_team, home_team)
    
    # Get actual scores and determine the actual winner
    actual_visitor_score = y_test_visitor.iloc[i]
    actual_home_score = y_test_home.iloc[i]
    actual_winner = visitor_team if actual_visitor_score > actual_home_score else home_team
    
    # Check if the prediction was correct
    if predicted_winner == actual_winner:
        correct_predictions += 1
    
    # Print game details
    print(f"Game {i+1}: {visitor_team} vs {home_team}")
    print(f"Predicted Score - {visitor_team}: {visitor_score}, {home_team}: {home_score}")
    print(f"Predicted Winner: {predicted_winner}")
    print(f"Actual Winner: {actual_winner}")
    print("-" * 50)

# Calculate and print the accuracy
accuracy = (correct_predictions / total_predictions) * 100
print(f"Prediction Accuracy: {accuracy:.2f}%")
