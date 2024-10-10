from itertools import groupby

import pandas as pd
import numpy as np
import random

# Adjust pandas display settings to show all columns
# pd.set_option('display.max_columns', None)  # Show all columns without truncation
# pd.set_option('display.max_rows', None)     # Optional: Show all rows
# pd.set_option('display.max_colwidth', None) # Optional: Show full width of each column


def get_avgs(team_data):
    teamName = team_data["team"]
    points = team_data["pts"].mean()
    fg_avg = team_data["fg"].mean()
    fg_avg_prob = team_data["fg%"].mean()
    threeP_avg = team_data["3p"].mean()
    threeP_avg_prob = team_data["3p%"].mean()
    twoP_avg_prob = team_data["2p%"].mean()
    ft_avg_prob = team_data["ft%"].mean()
    tov_avg_prob = team_data["tov_play"].mean()

    twoP_play_prob = team_data["2p_play"].mean()
    threeP_play_prob = team_data["3p_play"].mean()
    ft_play_prob = team_data["ft_play"].mean()

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
        "ft_play_prob": ft_play_prob,
    }


def game_possession(
    twop_play_chance,
    threep_play_chance,
    ft_play_chance,
    twop_prob,
    threep_prob,
    ft_prob,
    tov_prob,
):

    outcome = random.choices(
        ["2PT", "3PT", "FT", "TOV"],
        weights=[twop_play_chance, threep_play_chance, ft_play_chance, tov_prob],
        k=1,
    )[0]

    if outcome == "2PT":
        if random.random() < twop_prob:
            return 2
        else:
            return 0
    elif outcome == "3PT":
        if random.random() < threep_prob:
            return 3
        else:
            return 0
    elif outcome == "FT":
        if random.random() < ft_prob:
            return 2
        else:
            return 0
    elif outcome == "TOV":
        return 0


def simulate_game(teamA_avgs, teamB_avgs):

    # teamA_fg = np.random.normal(teamA_avgs['fg'],teamA_avgs['fg_prob'])

    teamA_score = 0

    teamA_2p_prob = teamA_avgs["twoP_avg_prob"]
    teamA_3p_prob = teamA_avgs["threeP_avg_prob"]
    teamA_tov_prob = teamA_avgs["tov_avg_prob"]
    teamA_ft_prob = teamA_avgs["ft_avg_prob"]
    teamA_2p_chance = teamA_avgs["twoP_play_prob"]
    teamA_3p_chance = teamA_avgs["threeP_play_prob"]
    teamA_ft_chance = teamA_avgs["ft_play_prob"]
    plays = 100

    for i in range(plays):
        teamA_score += game_possession(
            teamA_2p_chance,
            teamA_3p_chance,
            teamA_ft_chance,
            teamA_2p_prob,
            teamA_3p_prob,
            teamA_ft_prob,
            teamA_tov_prob,
        )
