import gzip
import json
import numpy as np
import os
import copy
import sys

def write_file(jsonfilename, data):
    with gzip.GzipFile(jsonfilename, 'w') as fout:
        fout.write(json.dumps(data).encode('utf-8'))


def process_file(jsonfilename):
    with gzip.GzipFile(jsonfilename, 'r') as fin:    # 4. gzip
        json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)

    json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
    data = json.loads(json_str)                      # 1. data
    return data


# names=["Churchton","Emmaus","Gravelly","Micanopy"]
# names = os.listdir('/home/dvruiz/externaldrive/pos/vribot/habitat-challenge-data/direct_policy_v6/')
names = os.listdir('habitat-challenge-data/objectnav_mp3d_v1/val/content/')
n_ep_per_class = 64


dict_mp3d = {'chair': 0, 'sofa': 0, 'picture': 0, 'table': 0, 'plant': 0, 'stool': 0, 'counter': 0, 'cabinet': 0, 'fireplace': 0, 'cushion': 0, 'bed': 0,
             'chest_of_drawers': 0, 'sink': 0, 'bathtub': 0, 'towel': 0, 'toilet': 0, 'seating': 0, 'shower': 0, 'gym_equipment': 0, 'tv_monitor': 0, 'clothes': 0}



# boolean_per_class_scene = {n:{} for n in names}
count_per_class = {}
for i, scene_name in enumerate(names):

    path='habitat-challenge-data/objectnav_mp3d_v1/val/content/'+scene_name
    # path='habitat-challenge-data/objectnav_mp3d_v1/train_small/content/'+scene_name

    print(path)
    v0= process_file(path)
    count_per_class_scene = {}
    # print(len(v0['episodes']))

    for ep in v0['episodes']:
        # print(ep)
        obj_class=ep['object_category']
        if(obj_class in count_per_class):
            count_per_class[obj_class]+=1
        else:
            count_per_class[obj_class]=1

        if(not(obj_class in count_per_class_scene)):
            dict_mp3d[obj_class]+=1
            count_per_class_scene[obj_class]=1

    #
print(count_per_class)

print(dict_mp3d)

# new_value = int(np.ceil( count_per_class_scene[key]/dict_mp3d[key] ))

# exit()
means = {}
for key in dict_mp3d:
    perc = n_ep_per_class/count_per_class[key]
    means[key] = int(np.ceil( (count_per_class[key]/dict_mp3d[key])*perc ))

print(means)
# new_value = int(np.ceil(n_ep_per_class/len(names)))
# print("new_value",new_value)

for i, scene_name in enumerate(names):

    path='habitat-challenge-data/objectnav_mp3d_v1/val/content/'+scene_name
    path_new='habitat-challenge-data/objectnav_mp3d_v1/test_mini/content/'+scene_name


    # print(path)
    v0= process_file(path)
    count_per_class_scene = {}
    ep_per_class = {}
    # new_episodes = np.array([])
    new_episodes = []

    for ep in v0['episodes']:
        obj_class=ep['object_category']
        if(obj_class in count_per_class_scene):
            count_per_class_scene[obj_class]+=1
            ep_per_class[obj_class].append(ep)
        else:
            count_per_class_scene[obj_class]=1
            ep_per_class[obj_class]=[ep]

    print("---------------------------------------------")
    total = 0
    for key in count_per_class_scene:
        # perc = count_per_class_scene[key]/count_per_class[key]
        # new_value = int(np.ceil(n_ep_per_class*perc))

        new_value = min(means[key],count_per_class_scene[key])
        print(key, new_value)
        total+=new_value

        # new_value = int(np.ceil( count_per_class_scene[key]/dict_mp3d[key] ))

        # new_value = int( np.floor( n_ep_per_class/(dict_mp3d[key]-1) ) )
        # new_value = int( np.floor( n_ep_per_class/(dict_mp3d[key]) ) )

        # print("new_value",new_value,"count_per_class_scene[key]",count_per_class_scene[key])
        # new_value_scene = min(count_per_class_scene[key], new_value)

        # perc = n_ep_per_class/count_per_class[key]
        # new_value = int(np.ceil(count_per_class_scene[key]*perc))

        # if(new_value>0):
        tmp =np.array(ep_per_class[key])
        np.random.shuffle(tmp)
        tmp = tmp[:new_value]
        # print(tmp.shape)
        tmp = tmp.tolist()
        new_episodes = new_episodes+tmp
        # new_episodes = [i for i in tmp]

    print("len(new_episodes)",len(new_episodes),"total",total)
    # v0['episodes']=new_episodes.tolist()
    v0['episodes']=new_episodes
    write_file(path_new, v0)

count_per_class = {}
for i, scene_name in enumerate(names):

    path='habitat-challenge-data/objectnav_mp3d_v1/test_mini/content/'+scene_name

    print(path)
    v0= process_file(path)
    count_per_class_scene = {}

    print(len(v0['episodes']))
    for ep in v0['episodes']:
        obj_class=ep['object_category']
        if(obj_class in count_per_class):
            count_per_class[obj_class]+=1
        else:
            count_per_class[obj_class]=1

        if(obj_class in count_per_class_scene):
            count_per_class_scene[obj_class]+=1
        else:
            count_per_class_scene[obj_class]=1
    #
    print(count_per_class_scene)
print(count_per_class)
