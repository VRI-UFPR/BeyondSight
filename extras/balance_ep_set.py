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

#folder path
# folder = '../habitat-challenge-data/mine_objectgoal_mp3d/lv2_test_small/content/'
# folder = '../habitat-challenge-data/mine_objectgoal_mp3d/train_small/content/'
# folder = '../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train/content/'
# folder = '../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train2/content/'
folder = '../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train3/content/'
# folder = '../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_test/content/'
# folder = '../habitat-challenge-data/objectgoal_mp3d/train/content/'
# folder = '../habitat-challenge-data/objectgoal_mp3d/train/content/'
#grab scenes filenames
filenames = os.listdir(folder)
#keep alphabetic ordering
filenames.sort()

#debug
print(filenames)

#keep a dict for ease of data walking
dataset_info = {}
total_count_per_class = {}

'''
For to extract data
'''
for i, file_name in enumerate(filenames):
    data = process_file(folder+file_name)
    data2 = process_file('../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train2/content/'+file_name)
    data['episodes']=data['episodes']+data2['episodes']
    #remove extension
    scene_name = file_name.split(".json.gz")[0]

    #debug
    print(scene_name)

    classes = list(data['category_to_task_category_id'].keys())
    #init with 0
    dataset_info[scene_name] = {key:[] for key in classes}


    for ep in data['episodes']:
        dataset_info[scene_name][ ep['object_category'] ].append(ep)

    # total = len(data['episodes'])
    desired_total_per_class = 100
    new_eps=[]
    for category in dataset_info[scene_name]:
        new_eps_cat=[]
        current = len(dataset_info[scene_name][category])
        print(category,current)
        perc = int(np.ceil(desired_total_per_class/current))
        for j in range(perc):
            new_eps_cat = new_eps_cat+dataset_info[scene_name][category]
        new_eps = new_eps + new_eps_cat[:100]

print(len(new_eps))
data['episodes']=new_eps
# data = json.dumps(data)
filename='../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train3_balanced/content/'+file_name
print("updating",filename,flush=True)
write_file(filename,data)
