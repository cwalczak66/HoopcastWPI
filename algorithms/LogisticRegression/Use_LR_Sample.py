from LR_Season_Input_Teams import *
home_team = 'GSW'
away_team = 'CLE'
predicted_winner, predicted_loser, home_prob, away_prob = predict_winner_for_teams(home_team, away_team)

print(f"Predicted winner: {predicted_winner}")
print(f"Predicted loser: {predicted_loser}")
print(f"{home_team} win probability: {home_prob}%")
print(f"{away_team} win probability: {away_prob}%")