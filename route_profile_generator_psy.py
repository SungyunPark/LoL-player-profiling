# from riotwatcher import LolWatcher, ApiError
import pandas as pd
import json
import cv2
import copy
import urllib.request
import json
import numpy as np
from pymongo import MongoClient
from replay_from_doc import ReplayConstructor
from refactoring.replay_construction import LoLReplayConstructor
import csv
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pylab as plt
from matplotlib import colors

csv_data = []
with open('./data/psy_jungle_300.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        csv_data.append(row[0].split(","))

# term
terms = []
for i in range(25//5):
    start = 51
    term = 10
    terms.append((start+(term*(i)), start+(term*(i+1))))

zero2fiv = (51, 61)
ten2twenty = (71, 91)
twen2twfi = (91, 101)
print(terms)

for i in range(len(terms)):
    print(terms[i])
    for j in range(*terms[i]):
        print(csv_data[0][j])

jungle_fp = []  # jungle footprints
for i in range(1, len(csv_data)):
    jungle_fp.append({})
    jungle_fp[-1]['name'] = csv_data[i][0]
    jungle_fp[-1]['fp'] = csv_data[i][51: 101]
    for j in range(len(terms)):
        tmp = str((terms[j][0]-terms[0][0])//2)
        tmp1 = str((terms[j][1]-terms[0][0])//2)
        index = tmp + '-' + tmp1
        tmp2 = 'fp_x_' + index
        tmp3 = 'fp_y_' + index
        tmp4 = [float(csv_data[i][k]) for k in range(*terms[j], 2)]
        term1 = terms[j][0]+1
        term2 = terms[j][1]
        tmp5 = [float(csv_data[i][k]) for k in range(term1, term2, 2)]
        jungle_fp[-1][tmp2] = tmp4
        jungle_fp[-1][tmp3] = tmp5
# Amount of data for 0-5 in csv_data will be gathered to loc_list['0-5']
# Amount of data for 5-10 in csv_data will be gathered to loc_list['5-10']
# Amount of data for 10-15 in csv_data will be gathered to loc_list['10-15']
# Amount of data for 15-20 in csv_data will be gathered to loc_list['15-20']
# Amount of data for 20-25 in csv_data will be gathered to loc_list['20-25']

loc_list = {}

for i in range(len(terms)):
    tmp1 = str((terms[i][0]-terms[0][0])//2)
    tmp2 = str((terms[i][1]-terms[0][0])//2)
    index = tmp1 + '-' + tmp2
    loc_list[index] = []
    for j in range(len(jungle_fp)):
        index = tmp1 + '-' + tmp2
        t5 = csv_data
        t6 = terms
        loc_list[index].append([float(t5[j+1][k]) for k in range(*t6[i])])
        jungle_fp[j]['fp_' + index] = loc_list[index][j]

K = range(1, 5)
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}

for i in range(len(loc_list)):
    tmp1 = str((terms[i][0]-terms[0][0])//2)
    tmp2 = str((terms[i][1]-terms[0][0])//2)
    index = tmp1 + '-' + tmp2
    X = np.array(loc_list[index])
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        tmp = cdist(X, kmeanModel.cluster_centers_, 'euclidean')
        tmp = np.min(tmp, axis=1)
        distortions.append(sum(tmp) / X.shape[0])
        inertias.append(kmeanModel.inertia_)
        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                       'euclidean'), axis=1)) / X.shape[0]
        mapping2[k] = kmeanModel.inertia_
    for key, val in mapping1.items():
        if key % 5 == 0:
            print(f'{key}: {val}')
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method in jungle footprint_' + index)
    plt.show()
    plt.close()
# get KMeans clusetering with 9 classes
# KMeans will be applied to loc_list data for each index 0-5, 5-10, 10-15,
# 15-20, 20-25
# each result of KMeans have same amount of csv_data
# kmeans is a list of the result of each KMeans
# jungle_fp, a list of dictionary, get another key class and is saved with the
# class value

kmeans = []
for i in range(len(terms)):
    num_clusters = 9
    tmp1 = str((terms[i][0]-terms[0][0])//2)
    tmp2 = str((terms[i][1]-terms[0][0])//2)
    index = tmp1 + '-' + tmp2
    tmp3 = loc_list[index]
    kmeans.append(KMeans(n_clusters=num_clusters, random_state=0).fit(tmp3))

for i in range(len(terms)):
    tmp1 = str((terms[i][0]-terms[0][0])//2)
    tmp2 = str((terms[i][1]-terms[0][0])//2)
    index = tmp1 + '-' + tmp2
    for j in range(len(jungle_fp)):
        jungle_fp[j]['class_' + index] = kmeans[i].labels_[j]


img = plt.imread('./data/map11.png')

c_list = list(range(0, num_clusters))
term = str(0) + '-' + str(10)
n_row = 3

for t in range(len(terms)):
    fig, axes = plt.subplots(n_row, n_row, figsize=(14, 14))
    tmp1 = str((terms[t][0]-terms[0][0])//2)
    tmp2 = str((terms[t][1]-terms[0][0])//2)
    index = tmp1 + '-' + tmp2
    tmp = str(len(jungle_fp))
    fig.suptitle(tmp + " Junglers footprint in " + index, fontsize=16)
    for c_num in c_list:
        num = 0
        r = c_num//n_row
        clmn = c_num % n_row
        axes[r, clmn].set_xlim([0, 1])
        axes[r, clmn].set_ylim([0, 1])
        axes[r, clmn].imshow(img, alpha=0.2, extent=[0, 1, 0, 1])
        for i in range(len(jungle_fp)):
            if jungle_fp[i]['class_'+index] == c_num:
                c = 'C' + str(jungle_fp[i]['class_' + index])
                st = 'incr_alpha'
                st1 = [(0, (*colors.to_rgb(c), 0)), (1, c)]
                cm = colors.LinearSegmentedColormap.from_list(st, st1)
                c_o = list(range(len(jungle_fp[i]['fp_x_'+index])))
                tmp_X = jungle_fp[i]['fp_x_'+index]
                tmp_Y = jungle_fp[i]['fp_y_'+index]
                axes[r, clmn].scatter(tmp_X, tmp_Y, c=c_o, cmap=cm, s=10**2)
                num += 1
        axes[r, clmn].set_title(str(num) + " players in class" + str(c_num))
    plt.show()
    plt.close()
    print("\n\n")
