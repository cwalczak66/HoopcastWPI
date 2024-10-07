

## [LR_Season_Input_Teams.py](LR_Season_Input_Teams.py): predict_winner_for_teams(season, team, team_opp) 

1. Call the `predict_winner_for_teams(season, team, team_opp)` function by passing:
   - `season`: The NBA season year (e.g., 2017).
   - `team`: The home team's code.
   - `team_opp`: The away team's code.
2. The function will return:
   - `predicted_winner`: The predicted winner of the game.
   - `predicted_loser`: The predicted loser of the game.
   - `home_team_prob`: The probability (in %) that the home team will win.
   - `away_team_prob`: The probability (in %) that the away team will win.

### Example

```python
from LR_Season_Input_Teams import *
home_team = 'GSW'
away_team = 'CLE'
predicted_winner, predicted_loser, home_prob, away_prob = predict_winner_for_teams(home_team, away_team)

print(f"Predicted winner: {predicted_winner}")
print(f"Predicted loser: {predicted_loser}")
print(f"{home_team} win probability: {home_prob}%")
print(f"{away_team} win probability: {away_prob}%")
