'''
Divide the dict json into chunks to run parallel generate_ep.py
'''

import numpy as np
import gzip
import json
import copy

from itertools import islice

def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

################################################################################
def process_file(jsonfilename):
    with gzip.GzipFile(jsonfilename, 'r') as fin:    # 4. gzip
        json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)

    json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
    data = json.loads(json_str)                      # 1. data
    return data

def write_file(jsonfilename, data):
    with gzip.GzipFile(jsonfilename, 'w') as fout:
        fout.write(data.encode('utf-8'))
################################################################################


# scenes_keys = list(scenes.keys())
#
# # scenes_keys = scenes_keys['ac26ZMwG7aT']
# for scene_name in scenes_keys:
#     if(scene_name!='ac26ZMwG7aT'):
#         del scenes[scene_name]
#
# print(scenes)
# scenes = json.dumps(scenes)
# fname = "../habitat-challenge-data/scenes_with_containing_classes_special_scene_with_all_classes.json.gz"
# write_file(fname, scenes)
# exit()
#
# scenes = process_file("../habitat-challenge-data/scenes_sorted_by_nonzero_instances_of_class.json.gz")
# n_classes = len(list(scenes.keys()))

scenes = process_file("../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train4/content/ac26ZMwG7aT.json.gz")
n_eps = len(scenes['episodes'])
chunk_size = int(np.ceil(n_eps / 5))

# mylist = [min(i,n_eps) for i in range(0,n_eps+chunk_size,chunk_size)]

for i in range(5):
    item  = scenes['episodes'][min((i)*chunk_size,n_eps):min((i+1)*chunk_size,n_eps)]

    # print(item,"\n")
    fname = "../habitat-challenge-data/mine_objectgoal_mp3d/single_scene_all_classes_train4/content/ac26ZMwG7aT_"+str(i)+".json.gz"
    # fname = "../habitat-challenge-data/scenes_sorted_by_nonzero_instances_of_class_"+str(i)+".json.gz"
    print(fname)
    tmp = scenes.copy()
    tmp['episodes'] = item
    tmp = json.dumps(tmp)
    write_file(fname, tmp)

print("i",i)

#
# scene_first = {}
#
# for class_name in scenes:
#     for scene_name in scenes[class_name]:
#         if scene_name in scene_first:
#             scene_first[scene_name].append(class_name)
#         else:
#             scene_first[scene_name] = [class_name]
#
# n_classes = len(list(scene_first.keys()))
# # fname = "../habitat-challenge-data/scenes_with_containing_classes.json.gz"
# # item = json.dumps(scene_first)
# # write_file(fname, item)
#
# # chunk_size = int(np.ceil(n_classes / 2))
# # chunk_size = int(np.ceil(n_classes / 4))
#
# # print("scene_first",scene_first)
#
# # max = 0
# # max_key = None
# #
# # for scene in scene_first:
# #     value = len(scene_first[scene])
# #     if(value>max):
# #         max = value
# #         max_key = scene
# #
# # print(max,max_key)
#
# # i = 0
# # for item in chunks(scene_first, chunk_size):
# #     print(item,"\n")
# #     fname = "../habitat-challenge-data/scenes_with_containing_classes_"+str(i)+"_of_2.json.gz"
# #     print(fname)
# #     item = json.dumps(item)
# #     write_file(fname, item)
# #     i+=1
# # #
# # print("i",i)
