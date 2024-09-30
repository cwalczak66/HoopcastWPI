import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import *

# Load the dataset
data = pd.read_csv('nba_games.csv')

# Drop unnecessary columns
# data = data.drop(['Date', 'Start (ET)', 'Attend.', 'LOG', 'Arena','Notes'], axis=1)

# Create target variable: 1 if home wins, 0 if visitor wins
# data['HomeWin'] = (data['HPTS'] > data['VPTS']).astype(int)

# Encode the team names using LabelEncoder
# label_encoder = LabelEncoder()
# data['Visitor/Neutral'] = label_encoder.fit_transform(data['Visitor/Neutral'])
# data['Home/Neutral'] = label_encoder.fit_transform(data['Home/Neutral'])

# # # Define the features (team names and points) and the target (HomeWin)
# # X = data[['Visitor/Neutral', 'VPTS', 'Home/Neutral', 'HPTS']]
# # y = data['HomeWin']

# # Define the features (team names) and the targets (VPTS, HPTS)
# X = data[['Visitor/Neutral', 'Home/Neutral']]
# y = data[['VPTS', 'HPTS']]

input_columns = [
    'mp', 'fg', 'fga', 'fg%', '3p', '3pa', '3p%', 'ft', 'fta', 'ft%', 
    'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 
    'ts%', 'efg%', '3par', 'ftr', 'orb%', 'drb%', 'trb%', 'ast%', 
    'stl%', 'blk%', 'tov%', 'usg%', 'ortg', 'drtg',
    'mp_opp', 'fg_opp', 'fga_opp', 'fg%_opp', '3p_opp', '3pa_opp', '3p%_opp', 
    'ft_opp', 'fta_opp', 'ft%_opp', 'orb_opp', 'drb_opp', 'trb_opp', 
    'ast_opp', 'stl_opp', 'blk_opp', 'tov_opp', 'pf_opp', 'pts_opp', 
    '+/-_opp', 'ts%_opp', 'efg%_opp', '3par_opp', 'ftr_opp', 'orb%_opp', 
    'drb%_opp', 'trb%_opp', 'ast%_opp', 'stl%_opp', 'blk%_opp', 'tov%_opp', 
    'usg%_opp', 'ortg_opp', 'drtg_opp', 
    'home'
]
# input_columns = [
#     '+/-', 'team_opp'
# ]
# Assuming you want to predict the points or game outcome:
target_column = 'won'  # Binary outcome, or 'pts' for points prediction

X = data[input_columns]  # Input features for training
y = data[target_column]  # Target variable (won or pts)


# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# # Build the ANN model
# model = Sequential()
# model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# Build the ANN model for regression
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))
# model.add(Dropout(0.2))  # Dropout layer with 50% dropout rate
# model.add(Dense(4096, activation='relu'))
# model.add(Dense(2048, activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2))  # 2 output neurons for VPTS and HPTS

# Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=SGD(learning_rate=0.0005), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model on the test set
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Test Accuracy: {accuracy * 100:.2f}%')
loss = model.evaluate(X_test, y_test)
print(f'Test Loss (MSE): {loss:.2f}')

# Make predictions
predictions = model.predict(X_test)
# predictions = (predictions > 0.5).astype(int)  # Convert probabilities to binary outcome (0 or 1)

# # Print a few predictions
# for i in range(10):
#     print(f"Predicted: {'Home Win' if predictions[i] == 1 else 'Away Win'}, Actual: {'Home Win' if y_test.iloc[i] == 1 else 'Away Win'}")

# Print a few predictions alongside actual scores
for i in range(10):
    print(f"Predicted Visitor Points: {predictions[i][0]:.2f}, Actual Visitor Points: {y_test.iloc[i, 0]}")
    print(f"Predicted Home Points: {predictions[i][1]:.2f}, Actual Home Points: {y_test.iloc[i, 1]}\n")



# Predict on test set
y_pred = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

plt.scatter(y_test['VPTS'], y_pred[:, 0], label='Visitor Points')
plt.scatter(y_test['HPTS'], y_pred[:, 1], label='Home Points', color='orange')
plt.plot([y_test.min().min(), y_test.max().max()], [y_test.min().min(), y_test.max().max()], 'r--')  # Diagonal line
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.legend()
plt.title('Actual vs Predicted Scores')
plt.show()