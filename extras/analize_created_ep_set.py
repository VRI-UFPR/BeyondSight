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
# folder = '../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train3/content/'
# folder = '../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train3_balanced/content/'

# folder = '../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_test/content/'
# folder = '../habitat-challenge-data/mine_objectgoal_mp3d/epset_starting_on_viewpoints/content/'
# folder = '../habitat-challenge-data/mine_objectgoal_mp3d/epset_starting_on_viewpoints_balanced/content/'
# folder = '../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train4/content/'
# folder = '../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train4_balanced/content/'
# folder = '../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train4_balanced_short_eps/content/'

# folder = '../habitat-challenge-data/objectgoal_mp3d/train/content/'
folder = '../habitat-challenge-data/objectgoal_mp3d/val_mini/'
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
    #remove extension
    scene_name = file_name.split(".json.gz")[0]

    #debug
    print(scene_name)

    classes = list(data['category_to_task_category_id'].keys())

    #init with 0
    dataset_info[scene_name] = {key:{'count':0,'geo_mean':0,'euc_mean':0,'geo_std':0,'euc_std':0,'geodesic_distance':[],'euclidean_distance':[],"path_len":[]} for key in classes}
    dataset_info[scene_name]['total_count'] = 0

    for ep in data['episodes']:
        dataset_info[scene_name][ ep['object_category'] ]['count'] += 1
        dataset_info[scene_name][ ep['object_category'] ]['geodesic_distance'].append( ep['info']['geodesic_distance'] )
        dataset_info[scene_name][ ep['object_category'] ]['euclidean_distance'].append( ep['info']['euclidean_distance'] )
        dataset_info[scene_name][ ep['object_category'] ]['path_len'].append( len(ep['shortest_paths'][0]) )

    for class_name in classes:
        if(dataset_info[scene_name][ class_name ]['count']!=0):
            dataset_info[scene_name][ class_name ]['geo_mean'] = np.mean(dataset_info[scene_name][ class_name ]['geodesic_distance'])
            dataset_info[scene_name][ class_name ]['euc_mean'] = np.mean(dataset_info[scene_name][ class_name ]['euclidean_distance'])

            dataset_info[scene_name][ class_name ]['geo_std'] = np.std(dataset_info[scene_name][ class_name ]['geodesic_distance'])
            dataset_info[scene_name][ class_name ]['euc_std'] = np.std(dataset_info[scene_name][ class_name ]['euclidean_distance'])

            dataset_info[scene_name][ class_name ]['path_mean'] = np.mean(dataset_info[scene_name][ class_name ]['path_len'])
            dataset_info[scene_name][ class_name ]['path_std'] = np.std(dataset_info[scene_name][ class_name ]['path_len'])

            #ONLY TO EASE OF VISUALIZATION
            del dataset_info[scene_name][ class_name ]['geodesic_distance']
            del dataset_info[scene_name][ class_name ]['euclidean_distance']
            del dataset_info[scene_name][ class_name ]['path_len']

            dataset_info[scene_name]['total_count']+=dataset_info[scene_name][ class_name ]['count']
            if class_name in total_count_per_class:
                total_count_per_class[class_name] += dataset_info[scene_name][ class_name ]['count']
            else:
                total_count_per_class[class_name] = dataset_info[scene_name][ class_name ]['count']
        else:
            del dataset_info[scene_name][ class_name ]
        ########################################################


print(dataset_info)
#
print("n_scenes",len(list(dataset_info.keys())))

print("\n\n",total_count_per_class)

min_class = np.inf
for key in total_count_per_class:
    if(total_count_per_class[key]<min_class):
        min_class=total_count_per_class[key]

print("\n\n",{key:min_class/total_count_per_class[key] for key in total_count_per_class})

total = 0
for key in total_count_per_class:
    total+=total_count_per_class[key]

print("total",total)

print("\n\n",{key:total_count_per_class[key]/total for key in total_count_per_class})
