#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


userMaxId = 0

def prepare_arrays_match(data,matR):

    for index,row in data.iterrows():
        
        user_id = row['user_id']
        hotel_cluster = row['hotel_cluster']
        
        if user_id != '':
            matR[int(user_id)][int(hotel_cluster)] += 1

def matrix_factorization_helper(R, P, Q, K, steps=500, alpha=0.2, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T

def matrix_factorization(matR):
    N = len(matR)
    M = len(matR[0])
    K = 10

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    nP, nQ = matrix_factorization_helper(matR, P, Q, K)
    nR = np.dot(nP, nQ.T)
    return nR

    
def find_score(nR,test):
    
    total = 0
    score = 0

    for index,row in test.iterrows():

        total += 1

        user_id = row['user_id']
        hotel_cluster = row['hotel_cluster']

        d = nR[user_id]
        topitems = np.argsort(-d)
        for i in range(5):
            if topitems[i] == hotel_cluster:
                score += 1.0 / (i+1)

    print('Result score:{}'.format(score*100.0/total))


def collaborative_filter(train,test):
    userMaxId = train['user_id'].max()
    matR = np.ones((userMaxId+1)*101, dtype='f').reshape(userMaxId+1, 101)
    prepare_arrays_match(train,matR)
    nR = matrix_factorization(matR)
    find_score(nR,test)