import pandas as pd
import numpy as np
import pybaseball as pb
import math
import os
import datetime as dt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from pybaseball import cache
from datetime import datetime
import joblib
import xgboost as xgb
cache.enable()

today = dt.date.today().strftime('YYYY-MM-DD')


selected_stats = [
    'Name', 'G', 'AB', 'PA', 'H', '2B', '3B', 'HR', 'R',
    'RBI', 'SB', 'CS', 'BB%', 'K%', 'OBP', 'SLG', 'wOBA',
    'xBA', 'xSLG', 'Barrels', 'EV', 'LA', 'WAR',
    'key_mlbam'
]

def get_season_data(start_dt, end_dt):
    """
    A script that queries 2022 statcast data week-by-week from opening day up to current date to handle api limits.

    start_dt: YYYY-MM-DD (opening day) str
    end_dt: YYYY-MM-DD (last day of the regular season) str
    """
    ## Searches for previously queried statcast data, if not found data is queried via pybaseball
    ## https://github.com/jldbc/pybaseball for more info

    df = pb.statcast(start_dt, end_dt)
    year = dt.datetime.strptime(start_dt, "%Y-%m-%d").year
    df.to_csv("statcast_data/{}.csv".format(year)) ## Saves statcast play-by-play data to .csv
    return df

def create_target_variable(df, events=['intentional_walk', 'walk', 'hit_by_pitch', 'single', 'double', 'triple', 'home_run', 'strikeout', 'other_out']):
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

def get_fg_stats(year, selected_stats=selected_stats):
    out = pb.fg_batting_data(year, qual=0)
    out = out.drop(columns=['xwOBA'])
    print(out.shape)
    id_table = pb.playerid_reverse_lookup(out['IDfg'], key_type='fangraphs')
    out = pd.merge(out, id_table, left_on='IDfg', right_on='key_fangraphs')

    return out[selected_stats]

def prepare_training_data(year):
    start_end_dates = pd.read_csv('start_end_dates.csv')

    csv_filename = f"statcast_data/{year}.csv"

    # Check if the CSV file for the corresponding year exists
    if os.path.exists(csv_filename):
        # If the CSV file exists, read the data from the CSV file
        df = pd.read_csv(csv_filename)
    else:
        start_dt, end_dt = start_end_dates.loc[start_end_dates['year'] == year, ['start_dt', 'end_dt']].values[0]
        df = get_season_data(start_dt, end_dt)

    sprint_speed = pd.read_csv("sprint_speed.csv")
    sprint_speed = sprint_speed[['player_id', 'sprint_speed']]
    df = pd.merge(df, sprint_speed, left_on='batter', right_on='player_id')
    return df

def preprocess_data(df):
    df = df.drop_duplicates()
    df['batter'] = df.batter.astype(int)
    df['game_year'] = pd.to_datetime(df['game_date']).dt.year

    stand_dummies = pd.get_dummies(df['stand'], prefix='stand')
    df = pd.concat([df, stand_dummies], axis=1)

    # Create a batter name column from mlbam ids
    names = pb.playerid_reverse_lookup(df['batter'])
    names['batter_name'] = names['name_first'] + " " + names['name_last']
    names = names[['key_mlbam', 'batter_name']]
    df = pd.merge(df, names, how='left', left_on='batter', right_on='key_mlbam')


    # Subset for batted ball events with non-null variables of interest
    bbe = df.loc[(df['description'] == 'hit_into_play') & (df['estimated_woba_using_speedangle'].notna())]
    bbe = bbe.dropna(subset=['launch_speed', 'launch_angle', 'stand_L', 'hc_x', 'hc_y'])

    non_bbe = df.loc[(df['estimated_woba_using_speedangle'].isna()) & (df['description'] != 'hit_into_play')]

    # [Rotate hit coordinates over 1st quadrant of xy-plane for ease in calculating spray angle]
    bbe['hc_x_adj'] = bbe['hc_x'].sub(126)
    bbe['hc_y_adj'] = 204.5 - bbe['hc_y']
    rad = -math.pi/4
    rotation_mat = np.array([[math.cos(rad), math.sin(rad)],
                            [-math.sin(rad), math.cos(rad)]])
    bbe[['field_x', 'field_y']] = bbe[['hc_x_adj', 'hc_y_adj']].dot(rotation_mat).astype(np.float64)

    ## Calculate spray angle (theta_deg) from inverse tangent function of transformed hit coordinates
    bbe[['field_x', 'field_y']] = bbe[['field_x', 'field_y']].astype(float)
    bbe['theta'] = np.arctan(bbe['field_y'].div(bbe['field_x']))
    bbe['theta_deg'] = bbe['theta'].mul(180/math.pi)

    labels = ['right', 'center', 'left']
    bins = pd.IntervalIndex.from_tuples([(bbe['theta_deg'].min(), 30), (30, 60), (60, bbe['theta_deg'].max())])
    bbe['hit_direction'] = pd.cut(bbe['theta_deg'], bins=bins).map(dict(zip(bins, labels)))

    bbe['pull'] = np.where(np.logical_or(np.logical_and((bbe['stand']=='R'),(bbe['hit_direction']=='left')), np.logical_and((bbe['stand']=='L'),(bbe['hit_direction']==1))), 1, 0)
    bbe['pulled_barrel'] = np.where(np.logical_and((bbe['pull']==1),(bbe['launch_speed_angle']==6)), 1, 0)

    # Create training dataframe for spray angle model
    bbe = bbe.dropna(subset=['launch_speed', 'launch_angle', 'woba_value'])
    bbe[['launch_speed', 'launch_angle', 'theta_deg']] = bbe[['launch_speed', 'launch_angle', 'theta_deg']].fillna(-100)

    return df, bbe


def train_model(bbe, year):
    X = bbe[['launch_speed', 'launch_angle', 'stand_L', 'sprint_speed']]
    y = bbe['woba_value'].values

    # Train a random forest regression model based on exit velocity and launch angle and using cross-validation to measure performance

    model1 = xgb.XGBRegressor(tree_method='gpu_hist')

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

    X_spray = bbe[['launch_speed', 'launch_angle', 'theta_deg', 'stand_L', 'sprint_speed']]
    y_spray = bbe['woba_value'].values

    

    try:
        model2 = joblib.load('models/rf_sxwoba_model.joblib')
    except:
        print("CV Spray angle xwOBA fit (R^2):") 
        print(cross_val_score(model2, X_spray, y_spray, cv=5))
        model2.fit(X_spray, y_spray)
    y_pred_spray = model2.predict(X_spray)

    bbe['sxwOBA'] = y_pred_spray

    # try:
    #     prob_model = joblib.load('models/prob_model.joblib')
    # except:
    prob_model.fit(X_spray, bbe['target'].values)
    probs = prob_model.predict_proba(X_spray)
    print(probs)
    bbe['sxwoba_probs'] = list(probs)
    lweights_2022 = np.array([0, 0.884, 1.261, 1.601, 2.072, 0])
    sxwoba = probs.dot(lweights_2022)
    bbe['sxwoba_prob_model'] = sxwoba
    bbe['sxwoba_prob_model'] = bbe['sxwoba_prob_model'].where(bbe['events']!='walk', 0.689)
    bbe['sxwoba_prob_model'] = bbe['sxwoba_prob_model'].where(bbe['events']!='hit_by_pitch', 0.720)
    print(prob_model.classes_)

    joblib.dump(model1, f'models/rf_xwoba_model_{year}.joblib')
    joblib.dump(model2, f'models/rf_sxwoba_model_{year}.joblib')
    joblib.dump(prob_model, f'models/prob_model_{year}.joblib')
    return bbe


def calculate_expected_xwoba(df, bbe, year):
    df['xwoba'] = df['woba_value']
    df['sxwoba'] = df['woba_value']
    

    df.loc[bbe.index, 'xwOBA'] = bbe['rf_xwoba']
    df.loc[bbe.index, 'sxwOBA'] = bbe['sxwOBA']
    df.loc[bbe.index, 'sxwoba_prob_model'] = bbe['sxwoba_prob_model']

    print(df.columns)
    df[['xwOBA', 'sxwOBA']] = df[['xwOBA', 'sxwOBA']].where(df['target'] != 0, 0.689)
    df[['xwOBA', 'sxwOBA']] = df[['xwOBA', 'sxwOBA']].where(df['target'] != 1, 0.689)
    df[['xwOBA', 'sxwOBA']] = df[['xwOBA', 'sxwOBA']].where(df['target'] != 2, 0.720)
    df[['xwOBA', 'sxwOBA']] = df[['xwOBA', 'sxwOBA']].where(df['target'] != 7, 0)

    leaders = df.loc[df['target']!=-1].groupby('batter')[['xwOBA', 'sxwOBA']].mean()
    sxwoba_leaders = df.loc[df['target'] != -1].groupby('batter')['sxwoba_prob_model'].agg(['mean', 'count'])
    return leaders, sxwoba_leaders

def postprocess_data(df, leaders, bbe, year):
    df['pulled_barrel'] = 0
    df.loc[bbe.index, 'pulled_barrel'] = bbe['pulled_barrel']
    pulled_barrels = df.groupby('batter')['pulled_barrel'].sum() 
    leaders['pulled_barrels'] = pulled_barrels
    cumulative_stats = get_fg_stats(year)

    leaders = leaders.merge(cumulative_stats, left_on='batter', right_on='key_mlbam')
    print('\n')
    print('leaders:', leaders.shape)
    print(cumulative_stats.loc[cumulative_stats['Name']=='David Villar'])

    # Data manipulation and inclusion of % diff 
    print(leaders.sort_values('PA', ascending=False).head(10))
    leaders['diff'] = leaders['sxwOBA'].sub(leaders['xwOBA'])
    leaders['diff%'] = leaders['diff'].div(leaders['xwOBA']).mul(100).round(2)
    print('\n')
    print(leaders.columns)

    leaders = leaders[['Name', 'key_mlbam', 'PA', 'wOBA', 'xwOBA', 'sxwOBA', 'diff', 'diff%', 'BB%', 'K%', 'HR', 'LA', 'Barrels', 'pulled_barrels']]
    leaders['BB%'] = leaders['BB%'].mul(100)
    leaders['K%'] = leaders['K%'].mul(100)
    leaders['year'] = year
    leaders = leaders.round(3)

    bbe['field_x'] = bbe.field_x.mul(2)
    bbe['field_y'] = bbe.field_y.mul(2)
    

    return df, leaders, bbe

def main():
    years_to_query = [2022, 2023]
    start_end = pd.read_csv("start_end_dates.csv")
    print(start_end.info())

    for year in years_to_query:
        season_df = prepare_training_data(year)
        season_df = create_target_variable(season_df)
        df, bbe = preprocess_data(season_df)
        bbe = train_model(bbe, year)
        leaders, sxwoba_leaders = calculate_expected_xwoba(df, bbe, year)
        df, leaders, bbe = postprocess_data(df, leaders, bbe, year)

        # Save data to csv
        bbe.to_csv(f"statcast_data/bbe_{year}.csv")
        leaders.to_csv(f"leaders/spray_xwoba_{year}.csv", index=False)


if __name__ == "__main__":
    main()