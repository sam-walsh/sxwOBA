def train_model(year, bbe, target='woba_value'):
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    X = bbe[['launch_speed', 'launch_angle', 'stand_L', 'sprint_speed']]
    y = bbe[target].to_numpy()
    

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
    if target == 'woba_value':
        try:
            model1 = joblib.load('models/rf_xwoba_model.joblib')

        except:
            print("CV xwOBA fit (R^2):") 
            print(cross_val_score(model1, X, y, cv=5))
            model1.fit(X, y)
    else:
        print("CV xwOBA fit (R^2):") 
        print(cross_val_score(model1, X, y, cv=5))
        model1.fit(X, y)
        

    y_pred = model1.predict(X)


    ## Grouping events by batter to get mean xwOBAcon for each player
    bbe['rf_xwoba'] = y_pred

    X_spray = bbe[['launch_speed', 'launch_angle', 'sprint_speed', 'pull', 'oppo']]
    y_spray = bbe[target].to_numpy()

    if target == 'woba_value':
        try:
                model2 = joblib.load('models/w.joblib')
        except:
            print("CV Spray angle xwOBA fit (R^2):") 
            print(cross_val_score(model2, X_spray, y_spray, cv=5))
            model2.fit(X_spray, y_spray)
    else:
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