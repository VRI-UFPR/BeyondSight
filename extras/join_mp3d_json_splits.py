import json

def process_first_json(split_a,mode, extra):
    filename_a= "../habitat-challenge-data/mp3d_2021_coco_style/"+extra+"/"+str(split_a)+"/"+mode+"/mp3d_2021_coco_style.json"
    with open(filename_a, 'r') as f:
        dict_a = json.load(f)

    for i in dict_a['images']:
        i['file_name']="./"+str(split_a)+"/"+mode+"/imgs/"+str(i['id'])+".jpg"

    return dict_a

def join_jsons(dict_a, split_b, mode, extra):

    #update image_id first so ann is consistent
    img_pad = len(dict_a['images'])+1
    ann_pad = len(dict_a['annotations'])+1

    filename_b= "../habitat-challenge-data/mp3d_2021_coco_style/"+extra+"/"+str(split_b)+"/"+mode+"/mp3d_2021_coco_style.json"
    with open(filename_b, 'r') as f:
        dict_b = json.load(f)

    for i in dict_b['images']:
        i['file_name']="./"+str(split_b)+"/"+mode+"/imgs/"+str(i['id'])+".jpg"
        i['id'] = i['id']+img_pad

    for i in dict_b['annotations']:
        i['id'] = i['id']+ann_pad
        i['image_id'] = i['image_id']+img_pad

    dict_ab = dict_a
    dict_ab['images']=dict_a['images']+dict_b['images']
    dict_ab['annotations']=dict_a['annotations']+dict_b['annotations']

    return dict_ab

################################################################################
# mode = "train"
mode = ""
extra = "viewpoints_train"
# mode = "val"
dict_0 = process_first_json(1,mode,extra)
print("done 0")
dict_01 = join_jsons(dict_0,2,mode,extra)
print("done 01")
dict_012 = join_jsons(dict_01,3,mode,extra)
print("done 012")
dict_0123 = join_jsons(dict_012,4,mode,extra)
print("done 0123")
dict_01234 = join_jsons(dict_0123,5,mode,extra)
print("done 01234")
################################################################################

mode = "train"
filename = "../habitat-challenge-data/mp3d_2021_coco_style/"+extra+"/mp3d_2021_coco_style_"+mode+".json"
print("saving",filename)
with open(filename, 'w') as f:
    # json.dump(dict_0123, f)
    json.dump(dict_01234, f)
