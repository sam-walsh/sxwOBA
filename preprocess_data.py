def create_target_variable(df, events=['intentional_walk', 'walk', 'hit_by_pitch', 'single', 'double', 'triple', 'home_run', 'strikeout', 'field_error', 'other_out']):
    df['target'] = -1
    df['des'] = df['des'].fillna('')
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

def get_fg_stats(year, selected_stats=[
    'Name', 'G', 'AB', 'PA', 'H', '2B', '3B', 'HR', 'R',
    'RBI', 'SB', 'CS', 'BB%', 'K%', 'OBP', 'SLG', 'wOBA',
    'xwOBA', 'xBA', 'xSLG', 'Barrels', 'EV', 'LA', 'WAR',
    'key_mlbam'
]
):
    import pybaseball as pb
    """
    returns fangraphs batting stats for a given year
    """
    out = pb.fg_batting_data(year, qual=1)
    out['first_name'] = out['Name'].str.split(n=1).str[0]
    out['last_name'] = out['Name'].str.split(n=1).str[1]
    out['key_mlbam'] = out.apply(get_mlbam_id, axis=1)
    out.dropna(subset=['key_mlbam'], inplace=True)
    return out[selected_stats]


def get_sprint_speed(df, year):
    """
    acquires sprint speed data for a given year and merges it with the statcast data
    """
    import pybaseball as pb
    import pandas as pd
    sprint_speed = pb.statcast_sprint_speed(year, min_opp=1)
    sprint_speed = sprint_speed[['player_id', 'sprint_speed']]
    df = pd.merge(df, sprint_speed, left_on='batter', right_on='player_id', how='left')
    df['sprint_speed'] = df['sprint_speed'].fillna(-1)
    df.drop(columns=['player_id'], inplace=True)
    return df

def impute_hit_direction_from_des(row):
    """
    Infer hit direction from the 'des' column.
    """
    import numpy as np
    import pandas as pd

    if 'left' in row['des']:
        return 'left'
    elif 'right' in row['des']:
        return 'right'
    elif 'center' in row['des']:
        return 'center'
    else:
        return np.nan

def preprocess_data(df):
    """
    Preprocesses the statcast data
    """
    import pandas as pd
    from sklearn.preprocessing import OrdinalEncoder
    import math
    import numpy as np
    import pybaseball as pb

    
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
    bbe = bbe.copy()


    # Rotate hit coordinates over 1st quadrant of xy-plane for ease in calculating spray angle
    bbe.loc[:, 'hc_x_adj'] = bbe['hc_x'].sub(126)
    bbe.loc[:, 'hc_y_adj'] = 204.5 - bbe['hc_y']
    rad = -math.pi/4
    rotation_mat = np.array([[math.cos(rad), math.sin(rad)],
                            [-math.sin(rad), math.cos(rad)]])
    
    bbe = bbe.dropna(subset=['hc_x', 'hc_y'])
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

    # Impute missing hit_direction values based on the 'des' column
    missing_hit_direction_mask = bbe['hit_direction'].isnull()
    bbe.loc[missing_hit_direction_mask, 'hit_direction'] = bbe.loc[missing_hit_direction_mask].apply(impute_hit_direction_from_des, axis=1)

    bbe['pull'] = bbe.apply(lambda row: (row['stand'] == 'R' and row['hit_direction'] == 'left') or (row['stand'] == 'L' and row['hit_direction'] == 'right'), axis=1).astype(int)
    bbe['oppo'] = bbe.apply(lambda row: (row['stand'] == 'R' and row['hit_direction'] == 'right') or (row['stand'] == 'L' and row['hit_direction'] == 'left'), axis=1).astype(int)

    # Creat pulled barrel indicator
    bbe['pulled_barrel'] = bbe.apply(lambda row: row['pull'] == 1 and row['launch_speed_angle'] == 6, axis=1).astype(int)

    # Create training dataframe for spray angle model
    bbe = bbe.dropna(subset=['launch_speed', 'launch_angle', 'hit_direction', 'woba_value'])
    
    return df, bbe