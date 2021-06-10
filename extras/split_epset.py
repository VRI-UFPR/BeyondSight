import os
import numpy as np

# importing shutil module
import shutil

folder = "../habitat-challenge-data/mine_objectgoal_mp3d/epset_starting_on_viewpoints_balanced/content/"
filepaths = os.listdir(folder)
# print("filepaths",filepaths)
# filepaths_size = [0 for i in range(len(filepaths))]
filepaths_size = np.zeros((len(filepaths)))
# Re-populate list with filename, size tuples
for i in range(len(filepaths)):
    filepaths_size[i] = os.path.getsize(folder+filepaths[i])

order = np.argsort(-filepaths_size)
filepaths = np.array(filepaths)
filepaths = filepaths[order]
# print("filepaths")


size = len(filepaths)
for i in range(0,size,5):

    if(i<size):
        current_split = 1
        folder_dst = "../habitat-challenge-data/mine_objectgoal_mp3d/epset_starting_on_viewpoints_balanced_split"+str(current_split)+"/content/"
        dest = shutil.copyfile(folder+filepaths[i], folder_dst+filepaths[i])
    if(i+1<size):
        current_split = 2
        folder_dst = "../habitat-challenge-data/mine_objectgoal_mp3d/epset_starting_on_viewpoints_balanced_split"+str(current_split)+"/content/"
        dest = shutil.copyfile(folder+filepaths[i+1], folder_dst+filepaths[i+1])
    if(i+2<size):
        current_split = 3
        folder_dst = "../habitat-challenge-data/mine_objectgoal_mp3d/epset_starting_on_viewpoints_balanced_split"+str(current_split)+"/content/"
        dest = shutil.copyfile(folder+filepaths[i+2], folder_dst+filepaths[i+2])
    if(i+3<size):
        current_split = 4
        folder_dst = "../habitat-challenge-data/mine_objectgoal_mp3d/epset_starting_on_viewpoints_balanced_split"+str(current_split)+"/content/"
        dest = shutil.copyfile(folder+filepaths[i+3], folder_dst+filepaths[i+3])
    if(i+4<size):
        current_split = 5
        folder_dst = "../habitat-challenge-data/mine_objectgoal_mp3d/epset_starting_on_viewpoints_balanced_split"+str(current_split)+"/content/"
        dest = shutil.copyfile(folder+filepaths[i+4], folder_dst+filepaths[i+4])
