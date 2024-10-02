from itertools import groupby

import pandas as pd
import numpy as np
import random
# Adjust pandas display settings to show all columns
# pd.set_option('display.max_columns', None)  # Show all columns without truncation
# pd.set_option('display.max_rows', None)     # Optional: Show all rows
# pd.set_option('display.max_colwidth', None) # Optional: Show full width of each column




def get_avgs(team_data):
    points = team_data['pts'].mean()
    fg_avg = team_data['fg'].mean()
    fg_avg_prob = team_data['fg%'].mean()
    threeP_avg = team_data['3p'].mean()
    threeP_avg_prob = team_data['3p%'].mean()





    return {
        "points": points,
        "fg_avg": fg_avg,
        "fg_avg_prob": fg_avg_prob,
        "threeP_avg": threeP_avg,
        "threeP_avg_prob": threeP_avg_prob

    }


def simulate_game(teamA_avgs,teamB_avgs):

    #teamA_fg = np.random.normal(teamA_avgs['fg'],teamA_avgs['fg_prob'])
    teamA_fg_prob = teamA_avgs['fg_avg_prob']
    teamA_3p_prob = teamA_avgs['threeP_avg_prob']

    print("fg prob:", teamA_fg_prob, "3p prob:", teamA_3p_prob)

    teamB_fg_prob = teamB_avgs['fg_avg_prob']
    teamB_3p_prob = teamB_avgs['threeP_avg_prob']

    teamA_shot_type = random.choices(['FG', '3PT'], weights=[teamA_fg_prob, teamA_3p_prob],k=1)[0]

    print(teamA_shot_type)



def monte_carlo(teamA,teamB,stats,sample_size):
    teamA_count = 0
    teamB_count = 0
    for i in range(sample_size):
        result = simulate_game(teamA,teamB,stats)
        if result == teamA:
            teamA_count += 1
        else:
            teamB_count += 1

    teamA_prob = teamA_count / sample_size
    teamB_prob = teamB_count / sample_size

    return teamA_prob, teamB_prob
#
#
#
#

def target_team(t):
    t["2p"] = t[""].shift(-1)
    return t

# # Defining main function
def main():

    # Load CSV data into a pandas DataFrame
    data = pd.read_csv('nba_games.csv')
    print(list(data.columns))

    # data_group = data.groupby('team', group_keys=False)[['team', 'won','pts', 'team_opp', 'pts_opp']].apply(target_team)

    team_code = data.pop('team')  # Remove 'Team Code' and save it
    data.insert(0, 'team', team_code)  # Insert 'Team Code' at the first position

    d1 = data[data["team"] == "WAS"]
    d2 = data[data["team"] == "BOS"]


    teamA = get_avgs(d1)
    teamB = get_avgs(d2)

    simulate_game(teamA,teamB)







    # print(data[data["team"] == "WAS"])


    #
    # # Display the first few rows to verify it's loaded correctly
    # teamA_points = data[['TeamA', 'PointsA']].rename(columns={"TeamA": "Team", "PointsA": "Points"})
    # teamB_points = data[['TeamB', 'PointsB']].rename(columns={"TeamB": "Team", "PointsB": "Points"})
    #
    #
    #
    # # # Combine both into one DataFrame
    # all_teams_points = pd.concat([teamA_points, teamB_points])
    #
    # team_stats = all_teams_points.groupby('Team')['Points'].agg(['mean', 'std'])
    #
    # print(team_stats)
    #
    # output = monte_carlo("Brooklyn Nets", "Chicago Bulls", team_stats, 10000)
    #
    # print(output)



if __name__ == "__main__":
    main()