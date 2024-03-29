def get_season_data(year):
    import pybaseball as pb
    import os
    import pandas as pd
    from datetime import datetime
    ## Searches for previously queried statcast data, if not found data is queried via pybaseball
    ## https://github.com/jldbc/pybaseball for more info

    start_end_dates = pd.read_csv(os.path.join('other_data', 'start_end_dates.csv'))

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

def get_fg_stats(year, selected_stats=[
    'Name', 'G', 'AB', 'PA', 'H', '2B', '3B', 'HR', 'R',
    'RBI', 'SB', 'CS', 'BB%', 'K%', 'OBP', 'SLG', 'wOBA',
    'xwOBA', 'xBA', 'xSLG', 'Barrels', 'EV', 'LA', 'WAR',
    'key_mlbam'
]
):
    import pybaseball as pb
    import pandas as pd
    out = pb.fg_batting_data(year, qual=1)
    id_table = pb.playerid_reverse_lookup(out['IDfg'].tolist(), key_type='fangraphs')
    out = pd.merge(out, id_table, left_on='IDfg', right_on='key_fangraphs')

    return out[selected_stats]