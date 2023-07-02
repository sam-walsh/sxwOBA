import pandas as pd
import numpy as np
import pybaseball as pb
import math
import os
import glob
import datetime as dt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from pybaseball import cache
from datetime import datetime
import joblib
import xgboost as xgb
from update_db import connect_to_db, create_tables, insert_data_into_tables, close_connection
from dotenv import load_dotenv
cache.enable()

today = dt.date.today().strftime('YYYY-MM-DD')


selected_stats = [
    'Name', 'G', 'AB', 'PA', 'H', '2B', '3B', 'HR', 'R',
    'RBI', 'SB', 'CS', 'BB%', 'K%', 'OBP', 'SLG', 'wOBA',
    'xwOBA', 'xBA', 'xSLG', 'Barrels', 'EV', 'LA', 'WAR',
    'key_mlbam'
]

def get_season_data(year):
    ## Searches for previously queried statcast data, if not found data is queried via pybaseball
    ## https://github.com/jldbc/pybaseball for more info

    start_end_dates = pd.read_csv('start_end_dates.csv')

    csv_filename = f"statcast_data/{year}.csv"

    # Check if the CSV file for the corresponding year exists
    if os.path.exists(csv_filename):
        # If the CSV file exists, read the data from the CSV file
        df = pd.read_csv(csv_filename, index_col=0)
        if df['game_date'].dtype == 'object':
            df['game_date'] = pd.to_datetime(df['game_date'])
        
        if year == 2023:
            max_date = df['game_date'].max()
            end_dt = pd.to_datetime(start_end_dates.loc[start_end_dates['year'] == year, 'end_dt'].values[0])

            if max_date + pd.DateOffset(days=1) < datetime.today():
                # Query for new data
                print((max_date + pd.DateOffset(days=1)).strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'))
                new_df = pb.statcast((max_date + pd.DateOffset(days=1)).strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'))

                
                # Append new data to the CSV if any new data was retrieved
                if not new_df.empty:
                    df = pd.concat([df, new_df]).reset_index(drop=True)
                    print(df.head())
    # Query from pybaseball using opening day and last day of the season
    else:
        start_dt, end_dt = start_end_dates.loc[start_end_dates['year'] == year, ['start_dt', 'end_dt']].values[0]
        df = pb.statcast(start_dt, end_dt)
    df.to_csv("statcast_data/{}.csv".format(year)) ## Saves statcast play-by-play data to .csv
    return df

def create_target_variable(df, events=['intentional_walk', 'walk', 'hit_by_pitch', 'single', 'double', 'triple', 'home_run', 'strikeout', 'field_error', 'other_out']):
    df['target'] = -1
    for i, event in enumerate(events):
        if event == 'intentional_walk':
            df.loc[df['des'].str.contains('intentional'), 'target'] = i
        elif event == 'walk':
            df.loc[df['events'] == event, 'target'] = i
            df.loc[(df['events'] == event) & (~df['des'].str.contains('intentional')), 'target'] = i
        elif event == 'strikeout':
            df.loc[df['events'] == 'strikeout', 'target'] = i
            df.loc[df['events'] == 'strikeout_double_play', 'target'] = i
        elif event == 'other_out':
            df.loc[df['events'].isin(['field_out', 'force_out', 'grounded_into_double_play', 'fielders_choice',
                                    'fielders_choice_out', 'double_play', 'triple_play', 'sac_fly', 'sac_fly_double_play']), 'target'] = i
        else:
            df.loc[df['events'] == event, 'target'] = i
    return df

def get_mlbam_id(row):
    """
    Attempts to find the mlbam id for a player using their fangraphs id and name
    """
    # First try fuzzy lookup using player first and last name
    result = pb.playerid_lookup(row['last_name'], row['first_name'], fuzzy=True)
    try:
        # If there are multiple results, try to find the one that matches the fangraphs id
        result = result.loc[result['key_fangraphs'] == row['IDfg']].iloc[0]
        return result['key_mlbam']
    except:
        # If no match, try reverse lookup using fangraphs id
        try:
            print('failed to find mlbam id for', row['Name'], row['last_name'], row['first_name'])
            print('trying reverse lookup')
            result = pb.playerid_reverse_lookup([row['IDfg']], key_type='fangraphs').iloc[0]
            return result['key_mlbam']
        except:
        # If no match, try to find the player with the same name that played in the mlb most recently
            result['mlb_played_last'].replace('', np.nan, inplace=True)
            result = result.loc[result['mlb_played_last'] > 2015].iloc[0]
            if len(result) != 0:
                return result['key_mlbam']
            else:
                return 000000

def get_sprint_speed(df, year):
    """
    acquires sprint speed data for a given year and merges it with the statcast data
    """
    sprint_speed = pb.statcast_sprint_speed(year, min_opp=1)
    sprint_speed = sprint_speed[['player_id', 'sprint_speed']]
    df = pd.merge(df, sprint_speed, left_on='batter', right_on='player_id', how='left')
    df['sprint_speed'] = df['sprint_speed'].fillna(-1)
    df.drop(columns=['player_id'], inplace=True)
    return df

def preprocess_data(df):
    """
    Preprocesses the statcast data
    """

    df = df.drop_duplicates()
    df['batter'] = df.batter.astype(int)

    # Handle missing shift data by assuming no shift
    df['if_fielding_alignment'] = df['if_fielding_alignment'].fillna('Standard')
    df['of_fielding_alignment'] = df['of_fielding_alignment'].fillna('Standard')
    encoder = OrdinalEncoder()

    # Fit and transform the problematic columns and convert it back to DataFrame
    encoded_data = encoder.fit_transform(df[['if_fielding_alignment', 'of_fielding_alignment']])
    encoded_df = pd.DataFrame(encoded_data, columns=['if_fielding_alignment', 'of_fielding_alignment'])

    # Join back with the original DataFrame
    df = df.drop(['if_fielding_alignment', 'of_fielding_alignment'], axis=1)
    df = df.join(encoded_df)

    # Create new columns
    df['batter_team'] = np.where(df['inning_topbot'] == 'Bot', df['home_team'], df['away_team'])
    df['pitcher_team'] = np.where(df['inning_topbot'] == 'Bot', df['away_team'], df['home_team'])
    df['game_year'] = pd.to_datetime(df['game_date']).dt.year

    stand_dummies = pd.get_dummies(df['stand'], prefix='stand')
    df = pd.concat([df, stand_dummies], axis=1)

    # Create a batter name column from mlbam ids
    names = pb.playerid_reverse_lookup(df['batter'].tolist())
    names['batter_name'] = names['name_first'] + " " + names['name_last']
    names = names[['key_mlbam', 'batter_name']].rename(columns={'key_mlbam': 'batter'})
    df = pd.merge(df, names, on='batter', how='left')

    # Subset for batted ball events with non-null variables of interest
    bbe = df.loc[(df['description'] == 'hit_into_play') & (df['estimated_woba_using_speedangle'].notna())]
    bbe = bbe.dropna(subset=['launch_speed', 'launch_angle', 'hc_x', 'hc_y'])

    # Rotate hit coordinates over 1st quadrant of xy-plane for ease in calculating spray angle
    bbe['hc_x_adj'] = bbe['hc_x'].sub(126)
    bbe['hc_y_adj'] = 204.5 - bbe['hc_y']
    rad = -math.pi/4
    rotation_mat = np.array([[math.cos(rad), math.sin(rad)],
                            [-math.sin(rad), math.cos(rad)]])
    
    bbe[['field_x', 'field_y']] = bbe[['hc_x_adj', 'hc_y_adj']].dot(rotation_mat).astype(np.float64)

    # Calculate spray angle (theta_deg) from inverse tangent function of transformed hit coordinates
    bbe[['field_x', 'field_y']] = bbe[['field_x', 'field_y']].astype(float)
    bbe['theta'] = np.arctan(bbe['field_y'].div(bbe['field_x']))
    bbe['theta_deg'] = bbe['theta'].mul(180/math.pi)
    bbe['theta_deg'] = bbe['theta_deg'].fillna(-1000)

    # Create categorical hit direction column and indicator pull/oppo columns
    labels = ['right', 'center', 'left']
    bins = pd.IntervalIndex.from_tuples([(bbe['theta_deg'].min(), 30), (30, 60), (60, bbe['theta_deg'].max())])
    bbe['hit_direction'] = pd.cut(bbe['theta_deg'], bins=bins).map(dict(zip(bins, labels)))

    bbe['pull'] = bbe.apply(lambda row: (row['stand'] == 'R' and row['hit_direction'] == 'left') or (row['stand'] == 'L' and row['hit_direction'] == 'right'), axis=1).astype(int)
    bbe['oppo'] = bbe.apply(lambda row: (row['stand'] == 'R' and row['hit_direction'] == 'right') or (row['stand'] == 'L' and row['hit_direction'] == 'left'), axis=1).astype(int)

    # Creat pulled barrel indicator
    bbe['pulled_barrel'] = bbe.apply(lambda row: row['pull'] == 1 and row['launch_speed_angle'] == 6, axis=1).astype(int)

    # Create training dataframe for spray angle model
    bbe = bbe.dropna(subset=['launch_speed', 'launch_angle', 'woba_value'])
    
    return df, bbe


def train_model(year, bbe):
     
    X = bbe[['launch_speed', 'launch_angle', 'stand_L', 'sprint_speed']]
    y = bbe['woba_value'].to_numpy()
    

    # Train a random forest regression model based on exit velocity and launch angle and using cross-validation to measure performance

    model1 = xgb.XGBRegressor(tree_method='gpu_hist',
                                n_estimators=100,
                                max_depth=6,
                                min_child_weight=1,
                                gamma=0,
                                subsample=1,
                                colsample_bytree=1)  

    model2 = xgb.XGBRegressor(tree_method='gpu_hist')

    prob_model = RandomForestClassifier(
        n_estimators=100
    )

    try:
        model1 = joblib.load('models/rf_xwoba_model.joblib')
    except:
        print("CV xwOBA fit (R^2):") 
        print(cross_val_score(model1, X, y, cv=5))
        model1.fit(X, y)

    y_pred = model1.predict(X)


    ## Grouping events by batter to get mean xwOBAcon for each player
    bbe['rf_xwoba'] = y_pred

    X_spray = bbe[['launch_speed', 'launch_angle', 'sprint_speed', 'pull', 'oppo']]
    y_spray = bbe['woba_value'].to_numpy()

    try:
        model2 = joblib.load('models/w.joblib')
    except:
        print("CV Spray angle xwOBA fit (R^2):") 
        print(cross_val_score(model2, X_spray, y_spray, cv=5))
        model2.fit(X_spray, y_spray)

    y_pred_spray = model2.predict(X_spray)

    bbe['sxwOBA'] = y_pred_spray

    prob_model.fit(X_spray, bbe['target'].to_numpy())
    probs = prob_model.predict_proba(X_spray)

    bbe['sxwoba_probs'] = list(probs)
    lweights = np.array([0, 0.9, 1.25, 1.6, 2, 0, 0.9])
    sxwoba = probs.dot(lweights)
    bbe['sxwoba_prob_model'] = sxwoba
    bbe['sxwoba_prob_model'] = bbe['sxwoba_prob_model'].where(bbe['events']!='walk', 0.689)
    bbe['sxwoba_prob_model'] = bbe['sxwoba_prob_model'].where(bbe['events']!='hit_by_pitch', 0.720)
    bbe['sxwoba_prob_model'] = bbe['sxwoba_prob_model'].where(bbe['events']!='strikeout', 0)

    return bbe


def calculate_expected_xwoba(df, bbe, year):
    df.loc[bbe.index, 'rf_xwoba'] = bbe['rf_xwoba']
    df.loc[bbe.index, 'sxwOBA'] = bbe['sxwOBA']
    df.loc[bbe.index, 'sxwoba_prob_model'] = bbe['sxwoba_prob_model']

    df[['rf_xwoba', 'sxwOBA']] = df[['rf_xwoba', 'sxwOBA']].mask(df['target'].isin([0, 1, 2]), 0.7) # wOBA for walks and HBP
    df[['rf_xwoba', 'sxwOBA']] = df[['rf_xwoba', 'sxwOBA']].mask(df['target'] == 7, 0) # wOBA for strikeouts

    woba_events = df.loc[df['target']!=-1]
    leaders = woba_events.groupby('batter')[['rf_xwoba', 'sxwOBA']].mean().reset_index()

    sxwoba_leaders = woba_events.groupby('batter')['sxwoba_prob_model'].agg(['mean', 'count'])
    return leaders, sxwoba_leaders

def postprocess_data(df, leaders, bbe, year):
    df['pulled_barrel'] = 0
    df.loc[bbe.index, 'pulled_barrel'] = bbe['pulled_barrel']
    pulled_barrels = df.groupby('batter')['pulled_barrel'].sum().reset_index()
    
    leaders = leaders.merge(pulled_barrels, on='batter').rename(columns={'pulled_barrel': 'pulled_barrels'})
    cumulative_stats = get_fg_stats(year).rename(columns={'key_mlbam': 'batter'})
    print(leaders.head())

    leaders = pd.merge(leaders, cumulative_stats, on='batter')
                            

    # Data manipulation and inclusion of % diff 
    print(leaders.sort_values('PA', ascending=False).head(10))
    leaders['diff'] = leaders['sxwOBA'].sub(leaders['xwOBA'])
    leaders['diff%'] = leaders['diff'].div(leaders['xwOBA']).mul(100).round(2)
    print(leaders.columns)

    leaders = leaders[['Name', 'batter', 'PA', 'wOBA', 'xwOBA', 'sxwOBA', 'diff', 'diff%', 'BB%', 'K%', 'HR', 'LA', 'Barrels', 'pulled_barrels']]
    leaders['BB%'] = leaders['BB%'].mul(100)
    leaders['K%'] = leaders['K%'].mul(100)
    leaders['year'] = year
    leaders = leaders.round(3) 

    bbe['field_x'] = bbe.field_x.mul(2)
    bbe['field_y'] = bbe.field_y.mul(2)
    

    return df, leaders, bbe

def main():
    years_to_query = [2021, 2022, 2023]
    start_end = pd.read_csv("start_end_dates.csv")

    for year in years_to_query:
        season_df = get_season_data(year)
        season_df = get_sprint_speed(season_df, year)
        season_df = create_target_variable(season_df)
        df, bbe = preprocess_data(season_df)
        bbe = train_model(year, bbe)
        leaders, sxwoba_leaders = calculate_expected_xwoba(df, bbe, year)
        df, leaders, bbe = postprocess_data(df, leaders, bbe, year)

        # Save data to csv
        bbe.to_csv(f"statcast_data/bbe/bbe_{year}.csv", index=False)
        leaders.to_csv(f"leaders/spray_xwoba_{year}.csv", index=False)
        if year == 2023:
            df.to_csv("statcast_data/current.csv")
    # Connect to the database
    conn, cur = connect_to_db()

    # Create the tables
    create_tables(cur)

    # Define the paths to your CSV files
    spray_xwoba_dir = 'C:/Users/wampl/sxwOBA/leaders/'
    bbe_dir = 'C:/Users/wampl/sxwOBA/statcast_data/bbe/'

    csv_file_paths_spray_xwoba = glob.glob(spray_xwoba_dir + '*.csv')
    csv_file_paths_bbe = glob.glob(bbe_dir + '*.csv')

    # Insert the data into the tables
    insert_data_into_tables(
        conn,
        cur,
        csv_file_paths_spray_xwoba,
        csv_file_paths_bbe
    )

    # Close the connection to the database
    close_connection(cur, conn)


if __name__ == "__main__":
    main()