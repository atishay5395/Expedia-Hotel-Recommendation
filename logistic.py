#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def generate_model(df):
    data_file = df
    model = df[['user_location_country','user_location_region','user_location_city','user_id','orig_destination_distance','is_mobile','srch_adults_cnt',
    'srch_children_cnt',
    'srch_rm_cnt',
    'srch_destination_type_id',
    'srch_destination_id',
    'hotel_continent',
    'is_booking',
    'hotel_country',
    'hotel_market','hotel_cluster',0,1,2]]
      
    y = model['hotel_cluster']
    model = model.drop(['hotel_cluster'], axis=1)
    scaler = StandardScaler()
    model = scaler.fit_transform(model)
    model_train, model_test, cluster_train, cluster_test = train_test_split(model, y, test_size=0.2, random_state=42)
    logit_model = LogisticRegression(penalty='l2')
    logit_model.fit(model_train, cluster_train)
    print ("Logistic regression is %2.2f" % accuracy_score(cluster_test, logit_model.predict(model_test)))
    
def init_logistic(data):
    generate_model(data)