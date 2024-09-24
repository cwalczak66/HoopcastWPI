import pandas as pd
import numpy as np




def simulate_game(teamA,teamB,stats):
    teamA_mean = stats.loc[teamA, "mean"]
    teamB_mean = stats.loc[teamB, "mean"]
    teamA_std = stats.loc[teamA, "std"]
    teamB_std = stats.loc[teamB, "std"]

    print()

    teamA_sim_pts = np.random.normal(teamA_mean, teamA_std)
    teamB_sim_pts = np.random.normal(teamB_mean, teamB_std)

    print(f"SIMULATION: {teamA}:", teamA_sim_pts, "{teamB}:", teamB_sim_pts)

    if teamA_sim_pts > teamB_sim_pts:
        return teamA
    else:
        return teamB


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




# Defining main function
def main():


    # Load CSV data into a pandas DataFrame
    data = pd.read_csv('TestData2.csv')

    print(data)

    # Display the first few rows to verify it's loaded correctly
    teamA_points = data[['TeamA', 'PointsA']].rename(columns={"TeamA": "Team", "PointsA": "Points"})
    teamB_points = data[['TeamB', 'PointsB']].rename(columns={"TeamB": "Team", "PointsB": "Points"})



    # # Combine both into one DataFrame
    all_teams_points = pd.concat([teamA_points, teamB_points])

    team_stats = all_teams_points.groupby('Team')['Points'].agg(['mean', 'std'])

    print(team_stats)

    output = monte_carlo("Boston Celtics", "Brooklyn Nets", team_stats, 100)

    print(output)



if __name__ == "__main__":
    main()