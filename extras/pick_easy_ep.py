'''
Extract metrics about the generated dataset
'''

################################################################################
##imports
import numpy as np
import gzip
import json

import os
import copy
import sys

################################################################################
##handle tar gjson
def write_file(jsonfilename, data):
    with gzip.GzipFile(jsonfilename, 'w') as fout:
        fout.write(json.dumps(data).encode('utf-8'))


def process_file(jsonfilename):
    with gzip.GzipFile(jsonfilename, 'r') as fin:    # 4. gzip
        json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)

    json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
    data = json.loads(json_str)                      # 1. data
    return data
################################################################################

file0 = '../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train/content/ac26ZMwG7aT.json.gz'
file1 = '../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train2/content/ac26ZMwG7aT.json.gz'
file2 = '../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_test/content/ac26ZMwG7aT.json.gz'
file3 = '../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_test2/content/ac26ZMwG7aT.json.gz'

# myfiles = [file0,file1,file2,file3]
#
# for file_curr in myfiles:
#     data = process_file(file_curr)

data0 = process_file(file0)
data1 = process_file(file1)
data2 = process_file(file2)
data3 = process_file(file3)
classes = list(data0['category_to_task_category_id'].keys())

new_episodes=[]
for class_name in classes:
    data0_ep=None
    data1_ep=None
    data2_ep=None
    data3_ep=None
    for id in range(len(data0['episodes'])):
        if (data0['episodes'][id]['object_category'] == class_name):
            data0_ep = data0['episodes'][id]
            break
    for id in range(len(data1['episodes'])):
        if (data1['episodes'][id]['object_category'] == class_name):
            data1_ep = data1['episodes'][id]
            break
    for id in range(len(data2['episodes'])):
        if (data2['episodes'][id]['object_category'] == class_name):
            data2_ep = data2['episodes'][id]
            break
    for id in range(len(data3['episodes'])):
        if (data3['episodes'][id]['object_category'] == class_name):
            data3_ep = data3['episodes'][id]
            break

    values = np.array([np.inf,np.inf,np.inf,np.inf])
    eps = [None,None,None,None]
    if(data0_ep):
        values[0]=len(data0_ep['shortest_paths'])
        eps[0]=data0_ep
    if(data1_ep):
        values[1]=len(data0_ep['shortest_paths'])
        eps[1]=data1_ep
    if(data2_ep):
        values[2]=len(data0_ep['shortest_paths'])
        eps[2]=data2_ep
    if(data3_ep):
        values[3]=len(data0_ep['shortest_paths'])
        eps[3]=data3_ep

    best = eps[np.argmin(values)]
    new_episodes.append(best)


data1['episodes'] = new_episodes
write_file(file1,data1)
