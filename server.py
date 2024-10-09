## TO RUN THIS FLASK SERVER
## RUN THE COMMAND BELOW
## python3 -m flask --app server run --debug

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import SelectField, SubmitField
from algorithms.poisson import poisson
from algorithms.logistic_regression import logistic_regression


app = Flask(__name__)
app.secret_key = "secret"
bootstrap = Bootstrap5(app)
csrf = CSRFProtect(app)

nba_teams = [
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "Los Angeles Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards",
]
nba_teams_abbrev = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


class TeamForm(FlaskForm):
    home_team = SelectField("Home Team", choices=[(team, team) for team in nba_teams])
    away_team = SelectField("Away Team", choices=[(team, team) for team in nba_teams])
    submit = SubmitField("Submit")


def algorithms(home_team, away_team):
    home_abr, away_abr = nba_teams_abbrev[home_team], nba_teams_abbrev[away_team]
    
    # Poisson
    poisson_home_score, poisson_away_score = poisson(home_team, away_team)
    poisson_str = f"{home_team} ({poisson_home_score} pts) vs {away_team} ({poisson_away_score} pts)"

    # Logistic Regression
    lr_predicted_winner, lr_predicted_loser, lr_home_prob, lr_away_prob = (
        logistic_regression(home_abr, away_abr)
    )
    lr_str = f"{home_team} ({round(lr_home_prob, 2)}%) vs {away_team} ({round(lr_away_prob, 2)}%)"
    
    

    return {"poisson": poisson_str, "logistic_regression": lr_str}


@app.route("/favicon.ico")
def favicon():
    return ""


@app.route("/", methods=["GET", "POST"])
def index():
    form = TeamForm()

    if form.validate_on_submit():
        home_team = form.home_team.data
        away_team = form.away_team.data

        if home_team == away_team:
            flash("You can't pick the same team for both teams.")
            return redirect(url_for("index"))

        return redirect(
            url_for(
                "predict",
                home=home_team,
                away=away_team,
                algs=algorithms(home_team, away_team),
            )
        )

    return render_template("index.html", form=form)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    home_team = request.args.get("home")
    away_team = request.args.get("away")

    return render_template(
        "predict.html",
        home_team=home_team,
        away_team=away_team,
        algs=algorithms(home_team, away_team),
    )
