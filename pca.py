#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
import pandas as pd

def perform_join(df,dest_small):
    
    ret = pd.DataFrame(df)
    #does a left join between the training data and the destination data
    #adds the 3 columns to the training data
    ret = ret.join(dest_small, on="srch_destination_id", how='left', rsuffix="dest")
    ret = ret.drop("srch_destination_iddest", axis=1)
    return ret

def init_pca(train):
    destinations = pd.read_csv("destinations.csv")
    #select 3 components out of 149
    pca = PCA(n_components=3)
    dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
    dest_small = pd.DataFrame(dest_small)
    dest_small["srch_destination_id"] = destinations["srch_destination_id"]
    df = perform_join(train,dest_small)
    return df