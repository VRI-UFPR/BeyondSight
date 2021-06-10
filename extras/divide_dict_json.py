'''
Divide the dict json into chunks to run parallel generate_ep.py
'''

import numpy as np
import gzip
import json

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
scenes = process_file("../habitat-challenge-data/scenes_with_containing_classes.json.gz")

scenes_keys = list(scenes.keys())

# scenes_keys = scenes_keys['ac26ZMwG7aT']
for scene_name in scenes_keys:
    if(scene_name!='ac26ZMwG7aT'):
        del scenes[scene_name]

print(scenes)
scenes = json.dumps(scenes)
fname = "../habitat-challenge-data/scenes_with_containing_classes_special_scene_with_all_classes.json.gz"
write_file(fname, scenes)
exit()

scenes = process_file("../habitat-challenge-data/scenes_sorted_by_nonzero_instances_of_class.json.gz")
n_classes = len(list(scenes.keys()))

# print("scenes",scenes)

# chunk_size = int(np.ceil(n_classes / 4))



#
# i = 0
# for item in chunks(scenes, chunk_size):
#     # print(item,"\n")
#     fname = "../habitat-challenge-data/scenes_sorted_by_nonzero_instances_of_class_"+str(i)+".json.gz"
#     print(fname)
#     item = json.dumps(item)
#     write_file(fname, item)
#     i+=1
#
# print("i",i)


scene_first = {}

for class_name in scenes:
    for scene_name in scenes[class_name]:
        if scene_name in scene_first:
            scene_first[scene_name].append(class_name)
        else:
            scene_first[scene_name] = [class_name]

n_classes = len(list(scene_first.keys()))
# fname = "../habitat-challenge-data/scenes_with_containing_classes.json.gz"
# item = json.dumps(scene_first)
# write_file(fname, item)

# chunk_size = int(np.ceil(n_classes / 2))
# chunk_size = int(np.ceil(n_classes / 4))

# print("scene_first",scene_first)

# max = 0
# max_key = None
#
# for scene in scene_first:
#     value = len(scene_first[scene])
#     if(value>max):
#         max = value
#         max_key = scene
#
# print(max,max_key)

# i = 0
# for item in chunks(scene_first, chunk_size):
#     print(item,"\n")
#     fname = "../habitat-challenge-data/scenes_with_containing_classes_"+str(i)+"_of_2.json.gz"
#     print(fname)
#     item = json.dumps(item)
#     write_file(fname, item)
#     i+=1
# #
# print("i",i)
