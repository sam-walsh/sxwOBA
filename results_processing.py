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
