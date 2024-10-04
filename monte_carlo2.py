from itertools import groupby

import pandas as pd
import numpy as np
import random
# Adjust pandas display settings to show all columns
# pd.set_option('display.max_columns', None)  # Show all columns without truncation
# pd.set_option('display.max_rows', None)     # Optional: Show all rows
# pd.set_option('display.max_colwidth', None) # Optional: Show full width of each column




def get_avgs(team_data):
    teamName = team_data['team']
    points = team_data['pts'].mean()
    fg_avg = team_data['fg'].mean()
    fg_avg_prob = team_data['fg%'].mean()
    threeP_avg = team_data['3p'].mean()
    threeP_avg_prob = team_data['3p%'].mean()
    twoP_avg_prob = team_data['2p%'].mean()
    ft_avg_prob = team_data['ft%'].mean()
    tov_avg_prob = team_data['tov_play'].mean()

    twoP_play_prob = team_data['2p_play'].mean()
    threeP_play_prob = team_data['3p_play'].mean()
    ft_play_prob = team_data['ft_play'].mean()






    return {
        "teamName": teamName,
        "points": points,
        "fg_avg": fg_avg,
        "fg_avg_prob": fg_avg_prob,
        "threeP_avg": threeP_avg,
        "twoP_avg_prob": twoP_avg_prob,
        "threeP_avg_prob": threeP_avg_prob,
        "ft_avg_prob": ft_avg_prob,
        "tov_avg_prob": tov_avg_prob,
        "twoP_play_prob": twoP_play_prob,
        "threeP_play_prob": threeP_play_prob,
        "ft_play_prob": ft_play_prob

    }

def game_possession(twop_play_chance, threep_play_chance, ft_play_chance, twop_prob, threep_prob, ft_prob, tov_prob):

    outcome = random.choices(['2PT', '3PT', 'FT','TOV'], weights=[twop_play_chance, threep_play_chance, ft_play_chance, tov_prob], k=1)[0]

    if outcome == '2PT':
        if random.random() < twop_prob:
            return 2
        else:
            return 0
    elif outcome == '3PT':
        if random.random() < threep_prob:
            return 3
        else:
            return 0
    elif outcome == 'FT':
        if random.random() < ft_prob:
            return 2
        else:
            return 0
    elif outcome == 'TOV':
        return 0




def simulate_game(teamA_avgs,teamB_avgs):

    #teamA_fg = np.random.normal(teamA_avgs['fg'],teamA_avgs['fg_prob'])

    teamA_score = 0


    teamA_2p_prob = teamA_avgs['twoP_avg_prob']
    teamA_3p_prob = teamA_avgs['threeP_avg_prob']
    teamA_tov_prob = teamA_avgs['tov_avg_prob']
    teamA_ft_prob = teamA_avgs['ft_avg_prob']
    teamA_2p_chance = teamA_avgs['twoP_play_prob']
    teamA_3p_chance = teamA_avgs['threeP_play_prob']
    teamA_ft_chance = teamA_avgs['ft_play_prob']
    plays = 100

    for i in range(plays):
        teamA_score += game_possession(teamA_2p_chance, teamA_3p_chance, teamA_ft_chance, teamA_2p_prob, teamA_3p_prob, teamA_ft_prob, teamA_tov_prob)



    teamB_score = 0

    teamB_2p_prob = teamB_avgs['twoP_avg_prob']
    teamB_3p_prob = teamB_avgs['threeP_avg_prob']
    teamB_tov_prob = teamB_avgs['tov_avg_prob']
    teamB_ft_prob = teamB_avgs['ft_avg_prob']
    teamB_2p_chance = teamB_avgs['twoP_play_prob']
    teamB_3p_chance = teamB_avgs['threeP_play_prob']
    teamB_ft_chance = teamB_avgs['ft_play_prob']
    plays = 100

    for i in range(plays):
        teamB_score += game_possession(teamB_2p_chance, teamB_3p_chance, teamB_ft_chance, teamB_2p_prob, teamB_3p_prob,
                                 teamB_ft_prob, teamB_tov_prob)


    if teamA_score > teamB_score:
        return "teamA"
    else:
        return "teamB"







def monte_carlo(teamA_avgs, teamB_avgs, sample_size):
    teamA_count = 0
    teamB_count = 0
    for i in range(sample_size):
        result = simulate_game(teamA_avgs, teamB_avgs)
        if result == "teamA":
            teamA_count += 1
        else:
            teamB_count += 1

    teamA_prob = teamA_count / sample_size
    teamB_prob = teamB_count / sample_size

    return teamA_prob, teamB_prob



# # Defining main function
def main():
    data2 = pd.read_csv('nba_games.csv')
    # Filter the dataset for the 2023 season
    filtered_data = data2[data2['season'] == '2022']

    df1 = data2[data2["season"] == 2022]





    data = pd.read_csv('nba_games.csv')




    # print("WELCOME TO NBA PREDICTION SIMULATION")
    # teamA_input = input("Select Team 1:")
    # teamB_input = input("Select Team 2:")
    # Load CSV data into a pandas DataFrame


    team_code = data.pop('team')  # Remove 'Team Code' and save it
    data.insert(0, 'team', team_code)  # Insert 'Team Code' at the first position

    data.loc[:, 'fg'] = data['fg'].fillna(0)
    data.loc[:, '3p'] = data['3p'].fillna(0)
    data.loc[:, 'fga'] = data['fga'].fillna(0)
    data.loc[:, '3pa'] = data['3pa'].fillna(0)

    data.loc[:, '2p'] = data['fg'] - data['3p']
    data.loc[:, '2pa'] = data['fga'] - data['3pa']
    data.loc[:, '2p%'] = data['2p'] / data['2pa']

    data.loc[:, 'total_goal_at'] = data['fta'] + data['3pa'] + data['2pa'] + data['tov']
    data.loc[:, '2p_play'] = data['2pa'] / data['total_goal_at']
    data.loc[:, '3p_play'] = data['3pa'] / data['total_goal_at']
    data.loc[:, 'ft_play'] = data['fta'] / data['total_goal_at']
    data.loc[:, 'tov_play'] = data['tov'] / data['total_goal_at']


    season_data = data[data['season'] == 2022]
    season_data_copy = data[data['season'] == 2022]
    season_data.reset_index(inplace=True)

    season_data = season_data.iloc[100:]




    first_100_rows = season_data_copy.iloc[:100]
    print(first_100_rows)
    team_info_list = first_100_rows[['team', 'team_opp', 'won']].apply(lambda row: (row['team'], row['team_opp'], row['won']),
                                                             axis=1).tolist()


    count = 0
    for game in team_info_list:
        teamA = game[0]
        teamB = game[1]
        game_result = game[2]
        print(game_result)

        d1 = season_data[season_data["team"] == teamA]
        d2 = season_data[season_data["team"] == teamB]


        teamA_avgs = get_avgs(d1)
        teamB_avgs = get_avgs(d2)

        # simulate_game(teamA,teamB)



        prediction = monte_carlo(teamA_avgs, teamB_avgs, 500)

        prediction_bool = True
        if prediction[0] < prediction[1]:
            prediction_bool = False

        if game_result and prediction_bool or not game_result and not prediction_bool:
            count += 1



        print(f"{teamA}", prediction[0], f"{teamB}", prediction[1])

    print(count, "%")



if __name__ == "__main__":
    main()