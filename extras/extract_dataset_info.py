'''
This script is intended to extract info from the jsons per scene giving a overall
dataset info object distribution of n of instances per class and per scene,
This data will be used to create the curriculum episode sets
'''
################################################################################
##imports
import gzip
import json
import numpy as np

import os
import copy
import sys

#debug
np.set_printoptions(threshold=sys.maxsize)
################################################################################

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
folder = '../habitat-challenge-data/objectgoal_mp3d/train/content/'
#grab scenes filenames
filenames = os.listdir(folder)
#keep alphabetic ordering
filenames.sort()

#debug
print(filenames)

#keep a dict for ease of data walking
dataset_info = {}
matrix = []

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
    dataset_info[scene_name] = {key:0 for key in classes}

    for j, class_name in enumerate(classes):
        class_with_pre_n_suffix = scene_name+".glb_"+class_name

        if class_with_pre_n_suffix in data['goals_by_category']:
            dataset_info[scene_name][class_name] = len(data['goals_by_category'][class_with_pre_n_suffix])

    #this give us an array of instances
    arr = list(dataset_info[scene_name].values())
    matrix.append(arr)

'''
Now we need to sort the scenes by instances
'''
matrix = np.array(matrix)

classes_size = matrix.shape[-1]
sorted_scenes_by_classes = [[] for idx in range(0, classes_size)]

class_names = np.array(list(data['category_to_task_category_id'].keys()))
scene_names = np.array(list(dataset_info.keys()))

for idx in range(0, classes_size):
    #idx th column only
    tmp = matrix[:,idx]
    sorted_idxs = np.argsort(tmp)
    sorted_scenes_by_classes[idx] = scene_names[sorted_idxs]

    #remove scenes with 0 instances
    sorted_scenes_by_classes[idx]= sorted_scenes_by_classes[idx][np.argwhere(tmp[sorted_idxs] != 0)]
    sorted_scenes_by_classes[idx]= sorted_scenes_by_classes[idx].reshape(sorted_scenes_by_classes[idx].shape[0])

#create dict and save it as json.gz
nice_dict = {key:sorted_scenes_by_classes[i].tolist() for i,key in enumerate(class_names)}
write_file("../habitat-challenge-data/scenes_sorted_by_nonzero_instances_of_class.json.gz",nice_dict)
