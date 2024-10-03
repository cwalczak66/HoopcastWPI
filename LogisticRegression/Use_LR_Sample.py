from LR_Season_Input_Teams import *
predicted_winner, predicted_loser, home_prob, away_prob = predict_winner_for_teams(2017, 'GSW', 'CLE')

print(f"Predicted winner: {predicted_winner}")
print(f"Home team win probability: {home_prob}%")
print(f"Away team win probability: {away_prob}%")