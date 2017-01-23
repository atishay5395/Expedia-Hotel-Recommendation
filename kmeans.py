#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
important features for k means:
    user_location_country
    user_location_region
    user_location_city
    orig_destination_distance
    is_mobile
    srch_adults_cnt
    srch_children_cnt
    srch_rm_cnt
    srch_destination_type_id
    srch_destination_id
    hotel_continent
    hotel_country
    hotel_market    
    
'''
import sys
import numpy as np
import random
import numpy


def assign_clusters(data, centroids):
    #assign the data to the nearest clusters
    clusters = {}
    cluster_key = 0
    for point in data:
        min_distance = sys.float_info.max
        length = len(centroids)
        new_point = point.copy()
        for i in range(0, length):
            center = centroids[i].copy()
            dist = np.sqrt(sum((np.asarray(new_point) - np.asarray(center)) ** 2))
            if dist <= min_distance:
                cluster_key = i
                min_distance = dist
        try:
            curr_cluster = clusters[cluster_key]
            curr_cluster.append(point)
        except:
            clusters[cluster_key] = [point]
    return clusters


def generate_new_centroids(clusters):
    #new centroids are formed by taking the means of the cluster points
    centroids = []
    keys = sorted(clusters.keys())
    for key in keys:
        centroids.append(np.mean(clusters[key], axis=0))
    return centroids

def centroids_converged(centroids, old_centroids, iterations):
    max_iterations = 200
    if iterations > max_iterations:
        return True
    return set([tuple(val) for val in centroids]) == set([tuple(val) for val in old_centroids])


def kmeans(data, k):
    #randomly select k cluster centers
    old_centroids = random.sample(list(data), k)
    centroids = random.sample(list(data), k)
    iterations = 0
    clusters = {}

    while not centroids_converged(centroids, old_centroids, iterations):
        iterations += 1
        old_centroids = centroids
        clusters = assign_clusters(data, centroids)
        centroids = generate_new_centroids(clusters)
    return centroids,clusters

        
def find_clusters(data, centroids):
    #store the distances in a heap
    heap = []
    score = 0
    for point in data:
        length = len(centroids)
        new_point = point.copy()
        hotel_cluster = new_point[14]
        new_point = np.delete(new_point,13)
        for i in range(0, length):
            center = centroids[i].copy()
            center = np.delete(center,13)
            dist = np.sqrt(sum((np.asarray(new_point) - np.asarray(center)) ** 2))
            heap.append(dist)
        #get the top 5 clusters with least distance
        ind = np.argpartition(heap,5)[:5]
        for i in range(len(ind)):
            c = int(centroids[i][14])
            if c == int(hotel_cluster):
                score += 1
                break
        heap = []
    print ("score:" ,score)


def init_kmeans(train,test,k):
    train.to_csv('sampletrain.csv') #n x 17

    inp_file = "sampletrain.csv"
    in_data = np.genfromtxt(inp_file, delimiter=',')
    in_data = numpy.delete(in_data,(0),axis = 0)
    centroids,clusters = (kmeans(data=in_data, k=k))
    
    test.to_csv('sampletest.csv')
    inp_file = "sampletest.csv"
    in_data = np.genfromtxt(inp_file, delimiter=',')
    in_data = numpy.delete(in_data,(0),axis = 0)


    find_clusters(in_data,centroids)
    return centroids,clusters