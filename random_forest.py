#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def feature_generation(data):
    data["date_time"] = pd.to_datetime(data["date_time"])
    data["srch_ci"] = pd.to_datetime(data["srch_ci"], format='%Y-%m-%d', errors="coerce")
    data["srch_co"] = pd.to_datetime(data["srch_co"], format='%Y-%m-%d', errors="coerce")
    
    props = {}
    for prop in ["month", "day", "hour", "minute", "dayofweek", "quarter"]:
        props[prop] = getattr(data["date_time"].dt, prop)
    
    carryover = [p for p in data.columns if p not in ["date_time", "srch_ci", "srch_co"]]
    for prop in carryover:
        props[prop] = data[prop]
    
    date_props = ["month", "day", "dayofweek", "quarter"]
    for prop in date_props:
        props["ci_{0}".format(prop)] = getattr(data["srch_ci"].dt, prop)
        props["co_{0}".format(prop)] = getattr(data["srch_co"].dt, prop)
    props["stay_span"] = (data["srch_co"] - data["srch_ci"]).astype('timedelta64[h]')
        
    ret = pd.DataFrame(props)
    return ret

def init_random_forest(df):
    df = feature_generation(df)
    df.fillna(-1, inplace=True)
    predictors = [c for c in df.columns if c not in ["hotel_cluster"]]
    from sklearn import cross_validation
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
    scores = cross_validation.cross_val_score(clf, df[predictors], df['hotel_cluster'], cv=3)
    print("The scores are:")
    
    print(scores)