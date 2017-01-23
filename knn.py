#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy import spatial
import numpy as np
import random
import sys
import operator
from collections import  defaultdict

#finds the closest cluster for each test data tuple
#euclidean distance
def calculate_closest_cluster(center_data, sample_data):
    closest_cluster =  defaultdict(list)
    for sdata in sample_data:
        min_dist = sys.float_info.max
        index = 0
        cnt = 0
        sample = sdata.copy()
        for cdata in center_data:
            center = cdata.copy()
            dist = np.sqrt(sum((np.asarray(sample) - np.asarray(center)) ** 2))
            if dist < min_dist:
                min_dist = dist
                index = cnt
            cnt += 1

        closest_cluster[index].append(sdata)

    return closest_cluster

#from each cluster, find the closest 5 neighbors based on the euclidean distance
def knn(cluster_data, sample_data):
    correct_count = 0
    for sample in sample_data:
        dist_dict = {}
        cnt = 0
        sam = sample.copy()
        for cluster in cluster_data:
            clu = cluster.copy()
            #compute the distance
            dist_dict[cnt] = np.sqrt(sum((np.asarray(sam) - np.asarray(clu)) ** 2))
            cnt += 1
        #sort the distance in ascending order
        dist_dict = sorted(dist_dict.items(), key=operator.itemgetter(1))
        
        #find the hotel clusters for the top 5 neighbors
        k = 0
        hotel_cluster_id = defaultdict(int)
        for key in dist_dict:
            if k == 5:
                break
            hotel_cluster_id[cluster_data[key[0]] [14]] += 1
            k += 1

        #sort according to the majority voting i.e. popularity of the hotel clusters
        hotel_cluster_id = sorted(hotel_cluster_id.items(), key=operator.itemgetter(1))
        if sample[14] == hotel_cluster_id[0][0]:
            correct_count += 1
    return correct_count


def init_knn(test,centroids,clusters):
    # Reading and storing input

    test.to_csv('sampletest.csv')
    inp_file = "sampletest.csv"
    sample_data = np.genfromtxt(inp_file, delimiter=',',dtype=int)
    sample_data = np.delete(sample_data,(0),axis = 0)
    
    center_data = centroids
    closest_cluster = calculate_closest_cluster(center_data, sample_data)
    correct = 0
    for key in closest_cluster:
        cluster_data = clusters[key]
        correct_count = knn(cluster_data, closest_cluster[key])
        correct += correct_count
    print ("-------------------")
    print (float(correct) / len(sample_data))  * 100