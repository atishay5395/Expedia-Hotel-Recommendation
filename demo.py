import pandas as pd
import operator
from collaborative_filtering import collaborative_filter
from kmeans import init_kmeans
from pca import init_pca
from knn import init_knn
from logistic import init_logistic
from random_forest import init_random_forest
import ml_metrics as metrics


def make_key(items):
    return "_".join([str(i) for i in items])

def assign_scores_to_clusters(data,test):
    match_cols = ["srch_destination_id"]
    cluster_cols = match_cols + ['hotel_cluster']
    #group by destination id and hotel cluster
    groups = data.groupby(cluster_cols)
    top_clusters = {}
    for name, group in groups:
        clicks = len(group.is_booking[group.is_booking == False])
        bookings = len(group.is_booking[group.is_booking == True])
        #assign 1 for each booking and 0.25 for each click
        score = bookings + .25 * clicks
    
        clus_name = make_key(name[:len(match_cols)])
        if clus_name not in top_clusters:
            top_clusters[clus_name] = {}
        top_clusters[clus_name][name[-1]] = score

    #for each destination id, find the top 5 clusters
    cluster_dict = {}
    for n in top_clusters:
        tc = top_clusters[n]
        top = [l[0] for l in sorted(tc.items(), key=operator.itemgetter(1), reverse=True)[:5]]
        cluster_dict[n] = top

    #read the test data, get the destination_id
    #predict from the top 5 hotel clusters
    preds = []
    for index, row in test.iterrows():

        key = str(int(row['srch_destination_id']))
        if key in cluster_dict:
            preds.append(cluster_dict[key])
        else:
            preds.append([])
    #compute the precision
    metrics.mapk([[l] for l in test["hotel_cluster"]], preds, k=5)

if __name__ == "__main__":
    # Reading and storing input
    rows = 200000
    train = pd.read_csv("train.csv",nrows = rows)

    #perform PCA to incorporate the destination latent features
    train = init_pca(train)
    
    #fill in the missing values with -1
    train.fillna(-1, inplace=True)
    
    #to perform k means, we decided to select only a subset of the features that had high
    #variance with respect to the hotel cluster
    train_kmeans = train[['user_location_country','user_location_region','user_location_city','orig_destination_distance','is_mobile','srch_adults_cnt',
    'srch_children_cnt',
    'srch_rm_cnt',
    'srch_destination_type_id',
    'srch_destination_id',
    'hotel_continent',
    'hotel_country',
    'hotel_market','hotel_cluster',0,1,2]]
    
    #the test data is selected from the training set to estimate the precision of the model
    test = pd.read_csv("train.csv",skiprows=range(1,rows),nrows=rows)
    test = init_pca(test)
    test.fillna(-1, inplace=True)
    test_kmeans = test[['user_location_country','user_location_region','user_location_city','orig_destination_distance','is_mobile','srch_adults_cnt',
    'srch_children_cnt',
    'srch_rm_cnt',
    'srch_destination_type_id',
    'srch_destination_id',
    'hotel_continent',
    'hotel_country',
    'hotel_market','hotel_cluster',0,1,2]]
    
    #logistic regression
    #init_logistic(train)
    
    #random forests
    #init_random_forest(train)
    
    #k means clustering
    centroids,clusters =  init_kmeans(train_kmeans,test_kmeans,k=150)
    #k nearest neighbors
    init_knn(test_kmeans,centroids,clusters)
    
    #Assign scores to clusters depending on the destination
    #assign_scores_to_clusters(train,test)

    #User based collaborative filtering
    #collaborative_filter(train,test)
    