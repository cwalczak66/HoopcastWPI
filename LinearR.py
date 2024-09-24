import pandas as pd
from sklearn.model_selection import train_test_split
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

# Split the data into training and testing sets
X_train, X_test, y_train_visitor, y_test_visitor, y_train_home, y_test_home = train_test_split(
    X, y_visitor, y_home, test_size=0.2, random_state=42)

# Train the linear regression model
model_visitor = LinearRegression()
model_home = LinearRegression()

model_visitor.fit(X_train, y_train_visitor)
model_home.fit(X_train, y_train_home)

# Function to predict the score and winner
def predict_winner(visitor_team, home_team):
    visitor_code = visitor_categories.get_loc(visitor_team)
    home_code = home_categories.get_loc(home_team)
    
    prediction_input = np.array([[visitor_code, home_code]])
    
    visitor_score = model_visitor.predict(prediction_input)[0]
    home_score = model_home.predict(prediction_input)[0]
    
    if visitor_score > home_score:
        winner = visitor_team
    else:
        winner = home_team
    
    return visitor_score, home_score, winner

# Example usage
visitor_team = 'Boston Celtics'
home_team = 'Denver Nuggets'
visitor_score, home_score, winner = predict_winner(visitor_team, home_team)

print(f"Predicted Score - {visitor_team}: {visitor_score}, {home_team}: {home_score}")
print(f"Predicted Winner: {winner}")