import pandas as pd
import numpy as np
import pybaseball as pb
import math
import os
import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from pybaseball import cache
cache.enable()

today = dt.date.today()

selected_stats = [
    'Name', 'G', 'AB', 'PA', 'H', '2B', '3B', 'HR', 'R',
    'RBI', 'SB', 'CS', 'BB%', 'K%', 'OBP', 'SLG', 'wOBA',
    'xwOBA', 'xBA', 'xSLG', 'Barrels', 'EV', 'LA', 'WAR',
    'key_mlbam'
]

def get_season_data():
    """
    A script that queries 2022 statcast data week-by-week from opening day up to current date to handle api limits.
    """

    ## Searches for previously queried statcast data, if not found data is queried via pybaseball
    ## https://github.com/jldbc/pybaseball for more info

    ## Divides query length into n queries of week length to handle api limits
    if len(os.listdir('statcast_data')) == 0:
        print("no statcast file found, querying 2022 data via pybaseball")

        weeks = []
        start = dt.date(2022, 4, 7)
        days = (today-start).days
        num_weeks = (days // 7) + 2
        counter = 0
        for d in range(days):
            if d%7 == 0:
                end = start + dt.timedelta(days=7)
                week = pb.statcast(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
                weeks.append(week)
                counter+=1
                print("week {}/{} complete".format(counter, num_weeks))
                start = end
                
            elif (d+1) == days:
                end = start + dt.timedelta(days=(d%7))
                week = pb.statcast(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
                counter+=1
                print("week {}/{} complete".format(counter, num_weeks))
            df = pd.concat(weeks)
            df.to_csv("statcast_data/{}.csv".format(today)) ## Saves statcast play-by-play data to .csv
        return df
    else:
        df = pd.read_csv('statcast_data/2022-10-18.csv')
        print("loading in saved statcast file")
        return df

def get_fg_stats(year, selected_stats=selected_stats):
    out = pb.fg_batting_data(year, qual=0)
    print(out.shape)
    id_table = pb.playerid_reverse_lookup(out['IDfg'], key_type='fangraphs')
    out = pd.merge(out, id_table, left_on='IDfg', right_on='key_fangraphs')

    return out[selected_stats]

def main():
    """
    A script that retrieves updated xwOBA and spray angle xwOBA leaderboards.
    """

    print("Overwrite previous statcast data file?")
    option = input('y/n: ')

    ## Retrieving 2022 statcast data via pybaseball
    df = get_season_data()
    df = df.drop_duplicates()
    df['batter'] = df.batter.astype(int)


    ## Creating a batter name column from mlbam ids
    names = pb.playerid_reverse_lookup(df['batter'])
    names['batter_name'] = names['name_first'] + " " + names['name_last']
    names = names[['key_mlbam', 'batter_name']]
    df = pd.merge(df, names, how='left', left_on='batter', right_on='key_mlbam')


    ## Subsetting for batted ball events with non-null variables of interest
    bbe = df.loc[(df['description'] == 'hit_into_play') & (df['estimated_woba_using_speedangle'].notna())]
    non_bbe = df.loc[(df['estimated_woba_using_speedangle'].isna()) & (df['description'] != 'hit_into_play')]

    print(bbe[['launch_speed', 'launch_angle', 'hc_x', 'hc_y', 'estimated_woba_using_speedangle']].isna().sum())
    print(bbe.loc[bbe['hc_x'].isna()])
    bbe = bbe.dropna(subset=['hc_x', 'hc_y'])
    

    ## Training a random forest regression model based on exit velocity and launch angle and using cross-validation to measure performance
    X = bbe[['launch_speed', 'launch_angle']]
    y = bbe['woba_value'].values

    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf = 100
    )

    model.fit(X, y)
    print("CV xwOBA fit (R^2):") 
    print(cross_val_score(model, X, y, cv=5))
    y_pred = model.predict(X)


    ## Grouping events by batter to get mean xwOBAcon for each player
    bbe['rf_xwoba'] = y_pred
    xwobacon_leaders = bbe.groupby(['batter_name', 'batter'])['rf_xwoba'].agg(['mean', 'count'])
    print('\n')
    print(xwobacon_leaders.shape)


    ## Counting all wOBA events (bbe, strikeouts, hbp, and BB) to count up total plate appearances
    num_pa = non_bbe.groupby(['batter_name', 'batter'])['woba_value'].agg(['mean', 'count'])
    print('\n')
    print(num_pa.shape)

    ## Calculating xwOBA from bbe xwOBAcon and non-bbe wOBA
    xwobacon_leaders['non_bbe'] = num_pa['mean']
    xwobacon_leaders['non_bbe_count'] = num_pa['count']
    xwobacon_leaders['total'] = num_pa['count'].add(xwobacon_leaders['count'])

    xwobacon_leaders['xwoba'] = ((xwobacon_leaders['mean']*xwobacon_leaders['count']) + \
                                (xwobacon_leaders['non_bbe']*xwobacon_leaders['non_bbe_count'])) / \
                                xwobacon_leaders['total']

    
    ## Rotating hit coordinates over 1st quadrant of xy-plane for ease in calculating spray angle
    bbe['hc_x_adj'] = bbe['hc_x'].sub(126)
    bbe['hc_y_adj'] = 204.5 - bbe['hc_y']
    rad = -math.pi/4
    rotation_mat = np.array([[math.cos(rad), math.sin(rad)],
                            [-math.sin(rad), math.cos(rad)]])
    bbe[['field_x', 'field_y']] = bbe[['hc_x_adj', 'hc_y_adj']].dot(rotation_mat).astype(np.float64)

    ## Calculating spray angle (theta_deg) from inverse tangent function of transformed hit coordinates
    bbe['field_x'] = bbe['field_x'].astype(float)
    bbe['field_y'] = bbe['field_y'].astype(float)
    bbe['theta'] = np.arctan(bbe['field_y'].div(bbe['field_x']))
    bbe['theta_deg'] = bbe['theta'].mul(180/math.pi)

    labels = ['right', 'center', 'left']
    bins = pd.IntervalIndex.from_tuples([(bbe['theta_deg'].min(), 30), (30, 60), (60, bbe['theta_deg'].max())])
    bbe['hit_direction'] = pd.cut(bbe['theta_deg'], bins=bins).map(dict(zip(bins, labels)))

    bbe['pull'] = np.where(((bbe['stand']=='R') & (bbe['hit_direction']=='left')) | ((bbe['stand']=='L') & (bbe['hit_direction']==1)), 1, 0)
    bbe['pulled_barrel'] = np.where(((bbe['pull']==1) & (bbe['launch_speed_angle']==6)), 1, 0)

    ## Incorporating spray angle into the random forest model
    bbe_spray = bbe.dropna(subset=['launch_speed', 'launch_angle', 'woba_value'])
    print(bbe_spray.columns)
    bbe_spray[['launch_speed', 'launch_angle', 'theta_deg']] = bbe_spray[['launch_speed', 'launch_angle', 'theta_deg']].fillna(0)

    X_spray = bbe_spray[['launch_speed', 'launch_angle', 'theta_deg']]
    y_spray = bbe_spray['woba_value'].values

    print("CV Spray angle xwOBA fit (R^2):") 
    print(cross_val_score(model, X_spray, y_spray, cv=5))

    model.fit(X_spray, y_spray)
    y_pred_spray = model.predict(X_spray)

    bbe['sxwOBA'] = y_pred_spray

    ## Generating spray angle xwOBA leaderboards
    spray_xwobacon = bbe.groupby(['batter_name', 'batter'])['sxwOBA'].agg(['mean', 'count'])
    print('\n')
    print(spray_xwobacon.shape)

    ## Adding spray angle xwOBA and pulled barrels to leaderboards
    pulled_barrels = bbe.groupby(['batter_name', 'batter'])['pulled_barrel'].agg(['mean', 'count'])
    xwobacon_leaders['pulled_barrels'] = pulled_barrels['count'].mul(pulled_barrels['mean']).round ()
    xwobacon_leaders['spray_xwoba_count'] = spray_xwobacon['count']
    xwobacon_leaders['sxwOBA'] = spray_xwobacon['mean']
    xwobacon_leaders = xwobacon_leaders.reset_index()
    print(xwobacon_leaders.head())

    ## Generating weighted average of bbe and non-bbe xwOBA and wOBA (cannot calculate xwOBA of non-bbe events)
    xwoba_sum = xwobacon_leaders['mean'].mul(xwobacon_leaders['spray_xwoba_count'])
    bbe_sum = xwobacon_leaders['sxwOBA'].mul(xwobacon_leaders['spray_xwoba_count'])
    non_bbe_sum = xwobacon_leaders['non_bbe'].mul(xwobacon_leaders['non_bbe_count'])
    total = xwobacon_leaders['non_bbe_count'].add(xwobacon_leaders['spray_xwoba_count'])
    spray_xwoba = (bbe_sum + non_bbe_sum) / total # spray angle xwOBA

    ## Creating data frame from leaders 
    cumulative_stats = get_fg_stats(2022)
    print('cumulative stats:', cumulative_stats.shape)
    spray_xwoba_leaders = pd.DataFrame.from_dict({
    'batter_name' : xwobacon_leaders['batter_name'],
    'batter_id': xwobacon_leaders['batter'],
    'sxwOBA': spray_xwoba,
    'pulled_barrels': xwobacon_leaders['pulled_barrels'],
    'bbe' : total
    })
    spray_xwoba_leaders = spray_xwoba_leaders.merge(cumulative_stats, left_on='batter_id', right_on='key_mlbam')
    print('\n')
    print('spray xwoba:', spray_xwoba_leaders.shape)
    print(cumulative_stats.loc[cumulative_stats['Name']=='David Villar'])
    

    ## Data manipulation and inclusion of % diff column
    spray_xwoba_leaders = spray_xwoba_leaders.loc[(spray_xwoba_leaders['xwOBA']!=0) & (spray_xwoba_leaders['sxwOBA']!=0)]  # prevents division by zero
    print(spray_xwoba_leaders.head())
    spray_xwoba_leaders['diff'] = spray_xwoba_leaders['sxwOBA'].sub(spray_xwoba_leaders['xwOBA'])
    spray_xwoba_leaders['diff %'] = spray_xwoba_leaders['diff'].div(spray_xwoba_leaders['xwOBA']).mul(100).round(2)
    print('\n')
    print(spray_xwoba_leaders.shape)

    ## Save leaderboards and bbe data to csv_file
    bbe.to_csv('bbe.csv')
    if option != 'y':
        spray_xwoba_leaders.to_csv("spray_xwoba{}.csv".format(today))
    else:
        spray_xwoba_leaders.to_csv("spray_xwoba.csv")

    return spray_xwoba_leaders

if __name__ == "__main__":
    main()
    