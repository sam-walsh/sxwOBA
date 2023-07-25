def postprocess_data(df, leaders, bbe, year):
    import pandas as pd
    import numpy
    from data_retrieval import get_fg_stats
    df['pulled_barrel'] = 0
    df.loc[bbe.index, 'pulled_barrel'] = bbe['pulled_barrel']
    pulled_barrels = df.groupby('batter')['pulled_barrel'].sum().reset_index()
    
    leaders = leaders.merge(pulled_barrels, on='batter').rename(columns={'pulled_barrel': 'pulled_barrels'})
    cumulative_stats = get_fg_stats(year).rename(columns={'key_mlbam': 'batter'})

    leaders = pd.merge(leaders, cumulative_stats, on='batter')
                            

    # Data manipulation and inclusion of % diff 
    print(leaders.sort_values('HR', ascending=False).head(10))
    leaders['diff'] = leaders['sxwOBA'].sub(leaders['xwOBA'])
    leaders['diff%'] = leaders['diff'].div(leaders['xwOBA']).mul(100).round(2)

    leaders = leaders[['Name', 'batter', 'PA', 'wOBA', 'xwOBA', 'sxwOBA', 'sxwOBA_adj', 'diff', 'diff%', 'BB%', 'K%', 'HR', 'LA', 'Barrels', 'pulled_barrels']]
    leaders['BB%'] = leaders['BB%'].mul(100)
    leaders['K%'] = leaders['K%'].mul(100)
    leaders['year'] = year
    leaders = leaders.round(3) 

    bbe['field_x'] = bbe.field_x.mul(2)
    bbe['field_y'] = bbe.field_y.mul(2)
    

    return df, leaders, bbe

def calculate_expected_xwoba(df, bbe, year):
    import pandas
    import numpy
    import joblib

    gidp_prob_model = joblib.load('models/gidp_prob_model.joblib')
    gidp_features = bbe['launch_angle'].values.reshape(-1, 1)
    gidp_prob = gidp_prob_model.predict_proba(gidp_features)[:, 1]
    bbe['gidp_prob'] = gidp_prob

    df.loc[bbe.index, 'rf_xwoba'] = bbe['rf_xwoba']
    df.loc[bbe.index, 'sxwOBA'] = bbe['sxwOBA']
    df.loc[bbe.index, 'sxwoba_prob_model'] = bbe['sxwoba_prob_model']
    df.loc[bbe.index, 'gidp_prob'] = bbe['gidp_prob']

    df[['rf_xwoba', 'sxwOBA', 'estimated_woba_using_speedangle']] = df[['rf_xwoba', 'sxwOBA', 'estimated_woba_using_speedangle']].mask(df['target'].isin([0, 1, 2]), 0.7) # wOBA for walks and HBP
    df[['rf_xwoba', 'sxwOBA', 'estimated_woba_using_speedangle']] = df[['rf_xwoba', 'sxwOBA', 'estimated_woba_using_speedangle']].mask(df['target'] == 7, 0) # wOBA for strikeouts

    woba_events = df.loc[df['target']!=-1]
    xwoba_mean = woba_events['estimated_woba_using_speedangle'].mean()
    print(xwoba_mean)
    woba_events.loc[:, 'gidp_adj'] = woba_events['gidp_prob'].mul(xwoba_mean).fillna(0)

    woba_events.loc[:, 'sxwOBA_adj'] = woba_events['sxwOBA'].sub(woba_events['gidp_adj'])
    woba_events.loc[:, 'xwOBA_adj'] = woba_events['estimated_woba_using_speedangle'].sub(woba_events['gidp_adj'])
    print(woba_events['sxwOBA_adj'].describe())
    if year in [2021, 2022, 2023]:
        woba_events.to_csv(f'statcast_data/woba_events_{year}.csv')
    leaders = woba_events.groupby('batter')[['rf_xwoba', 'sxwOBA', 'sxwOBA_adj']].mean().reset_index()

    # sxwoba_leaders = woba_events.groupby('batter')['sxwoba_prob_model'].agg(['mean', 'count'])
    return leaders
