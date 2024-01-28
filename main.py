import pandas as pd
import requests
from datetime import datetime
from autogluon.tabular import TabularDataset, TabularPredictor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Constants
TRAINING_YEARS = ["2019-20","2020-21","2021-22","2022-23"]
TEST_YEARS = ["2023-24"]
PLAYER_FILTER = 1 # (0 = Frequency, 1 = Points) – Frequency filters top players by frequency of appearance, points filters on sum of total points
NUMBER_PLAYERS_FILTER = 50 # This decides the number of players we filter by
NUMBER_GAMEWEEKS_AVG = 3 # This decides how many weeks back we want to look for LAST_N_FEATURES for a moving average
STANDARD_FEATURES = ['name', 'position', 'team', 'opponent_team', 'round', 'total_points', 'season', 'xP']
LAST_N_FEATURES = ['assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'expected_assists', 
                   'expected_goal_involvements', 'expected_goals', 'expected_goals_conceded', 'goals_conceded', 
                   'goals_scored', 'influence', 'minutes', 'own_goals', 'penalties_missed', 'penalties_saved', 
                   'red_cards', 'saves', 'selected', 'threat', 'total_points', 'transfers_balance', 
                   'transfers_in', 'transfers_out', 'value', 'yellow_cards', 'xP', 'total_points']

player_filters = {0:'frequency', 1:'points'}
label = 'total_points'
drop_columns = ['season', 'round', 'name', 'team', 'xP']
learner_kwargs = {'label_count_threshold': 1}

# Utility Functions
def get_teams(year):
    try:
        teams_url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{year}/teams.csv"
        teams = pd.read_csv(teams_url, encoding="utf-8")[["id", "name"]]
        teams.columns = ["opponent_team", "opponent"]
        return teams
    except Exception as e:
        logger.error(f"Error fetching teams for {year}: {e}")
        return pd.DataFrame()

def get_training_data(TRAINING_YEARS):
    list_of_gws = []
    for year in TRAINING_YEARS:
        for gameweek in range(1, 39):
            logger.info(f"Getting data for {year} gameweek {gameweek}")
            try:
                gw_data = pd.read_csv(f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{year}/gws/gw{gameweek}.csv", encoding="utf-8")
                
                # add opponent team
                teams = get_teams(year)
                gw_data = pd.merge(gw_data, teams, on="opponent_team", how="left")

                # add season
                gw_data["season"] = year
                list_of_gws.append(gw_data)

            except Exception as e:
                logger.error(f"Error fetching data for {year} gameweek {gameweek}: {e}")
    return pd.concat(list_of_gws)

def get_test_data(TEST_YEARS):
    list_of_gws = []
    for year in TEST_YEARS:
        for gameweek in range(1, 39):
            logger.info(f"Getting data for {year} gameweek {gameweek}")
            try:
                gw_data = pd.read_csv(f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{year}/gws/gw{gameweek}.csv", encoding="utf-8")
                
                # add opponent team
                teams = get_teams(year)
                gw_data = pd.merge(gw_data, teams, on="opponent_team", how="left")

                # add season
                gw_data["season"] = year
                list_of_gws.append(gw_data)

            except Exception as e:
                logger.error(f"Error fetching data for {year} gameweek {gameweek}: {e}")
    return pd.concat(list_of_gws)


def filter_players(all_data, PLAYER_FILTER):
    if PLAYER_FILTER == 1:
        player_points = all_data.groupby('name')['total_points'].sum()
        sorted_players = player_points.sort_values(ascending=False)
        top_players = sorted_players.head(NUMBER_PLAYERS_FILTER).index
        top_players_df = all_data[all_data['name'].isin(top_players)]
    else:
        player_frequencies = all_data['name'].value_counts()
        top_players = player_frequencies.head(NUMBER_PLAYERS_FILTER).index
        top_players_df = all_data[all_data['name'].isin(top_players)]
    return top_players_df

def calculate_features(all_data):
    number_of_lags = 3  # Number of weeks for lag features
    
    # Aggregate team-level metrics
    team_performance = all_data.groupby(['team', 'season', 'round']).agg({
        'goals_scored': 'sum',
        'goals_conceded': 'sum',
        'clean_sheets': 'sum',
        # Add other metrics as needed
    }).reset_index()

    # Rename the aggregated columns
    team_performance = team_performance.rename(columns={
        'goals_scored': 'team_goals_scored',
        'goals_conceded': 'team_goals_conceded',
        'clean_sheets': 'team_clean_sheets'
    })

    logger.info(list(team_performance.columns))
    logger.info(list(all_data.columns))
    
    # Create lag features for team performance
    for feature in ['team_goals_scored', 'team_goals_conceded', 'team_clean_sheets']:  # Add other team metrics as needed
        for lag in range(1, number_of_lags + 1):
            team_performance[f'{feature}_lag_{lag}'] = team_performance.groupby(['team', 'season'])[feature].shift(lag)

    # Merge team performance with main data
    all_data = pd.merge(all_data, team_performance, on=['team', 'season', 'round'], how='left')
    
    logger.info(list(team_performance.columns))
    logger.info(list(all_data.columns))

    # Calculate moving averages and lag features for individual players
    for feature in LAST_N_FEATURES:
        # Create lag features for the last 3 weeks
        for lag in range(1, number_of_lags + 1):
            all_data[f'{feature}_lag_{lag}'] = all_data.groupby(['name', 'season'])[feature].shift(lag)
        
        # Calculate mean and std dev of the last 3 weeks, lagged by 1 week
        all_data[f'{feature}_mean_last_3_lag'] = all_data[[f'{feature}_lag_{lag}' for lag in range(1, number_of_lags + 1)]].mean(axis=1).shift(1)
        all_data[f'{feature}_std_last_3_lag'] = all_data[[f'{feature}_lag_{lag}' for lag in range(1, number_of_lags + 1)]].std(axis=1).shift(1)

    # Select the relevant columns
    lag_feature_columns = [f'{feature}_lag_{lag}' for feature in LAST_N_FEATURES for lag in range(1, number_of_lags + 1)]
    lag_stat_columns = [f'{feature}_mean_last_3_lag' for feature in LAST_N_FEATURES] + [f'{feature}_std_last_3_lag' for feature in LAST_N_FEATURES]
    team_lag_feature_columns = [f'{feature}_lag_{lag}' for feature in ['goals_scored', 'goals_conceded', 'clean_sheets'] for lag in range(1, number_of_lags + 1)]
    
    return all_data[STANDARD_FEATURES + lag_feature_columns + lag_stat_columns + team_lag_feature_columns]

def count_nans(final_data):
    na_count = final_data.isna().sum()
    with pd.option_context('display.max_rows', None,
            'display.max_columns', None,
            'display.precision', 3):
        logger.info(na_count)

def drop_nans(final_data):
    final_data = final_data.dropna(ignore_index = True)
    return final_data

def train_model(final_data):
    train = final_data.sample(frac=0.8, random_state=200)
    test = final_data.drop(train.index)

    train_x = train.drop(drop_columns, axis=1)
    train_y = train['total_points']

    train_data = TabularDataset(train_x)
    predictor = TabularPredictor(label=label, learner_kwargs=learner_kwargs).fit(train_data)
    predictor.evaluate(test, display=True)
    predictor.leaderboard(test, display=True)
    return predictor

# Future utility functions
def get_current_gw():
    """Get's the most recent gameweek's ID."""
    try:
        data = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/').json()
        now = datetime.utcnow()
        for gameweek in data['events']:
            next_deadline_date = datetime.strptime(gameweek['deadline_time'], '%Y-%m-%dT%H:%M:%SZ')
            if next_deadline_date > now:
                return gameweek['id'] - 1
    except requests.RequestException as e:
        logger.error(f"Error fetching current gameweek: {e}")
        return None


# Main
if __name__ == "__main__":
    all_data = filter_players(get_training_data(TRAINING_YEARS), PLAYER_FILTER).reset_index(drop=True)
    all_data.to_csv("gw_rows.csv")

    final_data = calculate_features(all_data).reset_index(drop=True)
    count_nans(final_data)
    # final_data = drop_nans(final_data).reset_index(drop=True)
    final_data.to_csv(f"top{NUMBER_PLAYERS_FILTER}_lag1_last{NUMBER_GAMEWEEKS_AVG}_{player_filters[PLAYER_FILTER]}.csv")

    test_data = filter_players(get_test_data(TEST_YEARS), PLAYER_FILTER).reset_index(drop=True)
    test_data = calculate_features(test_data).reset_index(drop=True)
    count_nans(test_data)
    # test_data = drop_nans(test_data).reset_index(drop=True)
    test_data.to_csv(f"test_data.csv")