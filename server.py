## TO RUN THIS FLASK SERVER
## RUN THE COMMAND BELOW
## python3 -m flask --app server run --debug

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import SelectField, SubmitField

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

class TeamForm(FlaskForm):
    home_team = SelectField("Home Team", choices=[(team, team) for team in nba_teams])
    away_team = SelectField("Away Team", choices=[(team, team) for team in nba_teams])
    submit = SubmitField("Submit")

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
            flash("You can\'t pick the same team for both teams.")
            return redirect(url_for("index"))
        
        flash(f"{home_team} vs {away_team}")
        return redirect(url_for("index"))
    
    return render_template("index.html", form=form)
