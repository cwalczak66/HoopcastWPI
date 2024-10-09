# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:02:05 2024

CS4341: Introduction to Artificial Intelligence
Final Project
Predicitive Model Using Random Forest Classifier

@author: ethan
"""

import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

"""
Fuctions to run the model
"""


# adds a column for what we are predicting (next game)
def add_target(team):
    team["target"] = team["won"].shift(-1)
    return team


# split data and use historical data to predict future
def backtest(data, model, predictors, start=2, step=1):
    all_predictions = []

    seasons = sorted(data["season"].unique())

    for i in range(start, len(seasons), step):
        season = seasons[i]
        train = data[data["season"] < season]
        test = data[data["season"] == season]

        model.fit(train[predictors], train["target"])

        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]

        all_predictions.append(combined)
    return pd.concat(all_predictions)


# use rolling averages to improve the model
def find_team_averages(team, num_games=10):
    rolling = team.rolling(num_games).mean()
    return rolling


# shift columns to add previously known info
def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col


def add_col(df, col_name):
    return df.groupby("team", group_keys=False).apply(
        lambda x: shift_col(x, col_name), include_groups=False
    )


def random_forest_classifier():
    """
    Preparing Data for Model
    """

    # read in data to file (data aquired from DataQuest Machine Learning Tutorial)
    # sort data by data and reindex
    data = pd.read_csv("data/nba_games.csv").sort_values("date").reset_index(drop=True)

    # remove extra columns
    del data["Unnamed: 0"]
    del data["mp.1"]
    del data["mp_opp.1"]
    del data["index_opp"]

    # group data by team
    data = data.groupby("team", group_keys=False).apply(add_target)

    # remove null values from target column (after last game of the season)
    data.loc[pd.isnull(data["target"]), "target"] = 2
    data["target"] = data["target"].astype(int, errors="ignore")

    # remove columns that contain null values
    nulls = pd.isnull(data).sum()
    nulls = nulls[nulls > 0]

    valid_columns = data.columns[~data.columns.isin(nulls.index)]

    data = data[valid_columns].copy()

    """
    Machine Learning Model
    """

    # initialize sklearn classes

    rr = RidgeClassifier(alpha=1)
    rf = RandomForestClassifier(100)
    split = TimeSeriesSplit(n_splits=3)

    sfs = SequentialFeatureSelector(
        rr, n_features_to_select=30, direction="forward", cv=split
    )

    # select columns for scaling the data (all values fall between 0-1)
    removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
    selected_columns = data.columns[~data.columns.isin(removed_columns)]

    scaler = MinMaxScaler()
    data[selected_columns] = scaler.fit_transform(data[selected_columns])

    # fit sequential feature selector
    sfs.fit(data[selected_columns], data["target"])

    # get list of used stats
    predictors = list(selected_columns[sfs.get_support()])

    # create rolling averages
    data_rolling = data[list(selected_columns) + ["won", "team", "season"]]
    data_rolling = data_rolling.groupby(["team", "season"], group_keys=False).apply(
        find_team_averages, include_groups=False
    )

    # add rolling averages to data dataframe
    rolling_cols = [f"{col}_10" for col in data_rolling.columns]
    data_rolling.columns = rolling_cols
    data = pd.concat([data, data_rolling], axis=1)

    # drop all null rows
    data = data.dropna()

    # add known info to model
    data["home_next"] = add_col(data, "home")
    data["team_opp_next"] = add_col(data, "team_opp")
    data["date_next"] = add_col(data, "date")

    # remove pointers to slices
    data = data.copy()

    # add info about the opponent
    full = data.merge(
        data[rolling_cols + ["team_opp_next", "date_next", "team"]],
        left_on=["team", "date_next"],
        right_on=["team_opp_next", "date_next"],
    )

    # get a new list of removed and selected columns
    removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns

    selected_columns = full.columns[~full.columns.isin(removed_columns)]

    sfs.fit(full[selected_columns], full["target"])

    # get new list of selected columns
    predictors = list(selected_columns[sfs.get_support()])

    # run new model
    predictions = backtest(full, rr, predictors)
    accuracy = accuracy_score(predictions["actual"], predictions["prediction"])
    
    return accuracy