import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import utils
# from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

import habitat_sim.registry as registry
from habitat_sim.utils.data import ImageExtractor, PoseExtractor

import pycocotools.mask
import json
from PIL import Image

import os

# Replace with the path to your scene file
# SCENE_FILEPATH = 'data/scene_datasets/habitat-test-scenes/apartment_0/mesh.ply'
# SCENE_FILEPATHS = ['/habitat-challenge-data/data/scene_datasets/mp3d/ac26ZMwG7aT/ac26ZMwG7aT.glb']

SCENE_FILEPATHS = [
    "/habitat-challenge-data/data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/ac26ZMwG7aT/ac26ZMwG7aT.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/x8F5xyUWy9e/x8F5xyUWy9e.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/1pXnuDYAj8r/1pXnuDYAj8r.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/29hnd4uzFmX/29hnd4uzFmX.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/2n8kARJN3HM/2n8kARJN3HM.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/2t7WUuJeko7/2t7WUuJeko7.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/5LpN3gDmAk7/5LpN3gDmAk7.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/5q7pvUzZiYa/5q7pvUzZiYa.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/5ZKStnWn8Zo/5ZKStnWn8Zo.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/759xd9YjKW5/759xd9YjKW5.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/7y3sRwLe3Va/7y3sRwLe3Va.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/8194nk5LbLH/8194nk5LbLH.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/82sE5b5pLXE/82sE5b5pLXE.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/8WUmhLawc2A/8WUmhLawc2A.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/aayBHfsNo7d/aayBHfsNo7d.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/ARNzJeq3xxb/ARNzJeq3xxb.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/B6ByNegPMKs/B6ByNegPMKs.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/b8cTxDM8gDG/b8cTxDM8gDG.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/cV4RVeZvu5T/cV4RVeZvu5T.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/D7G3Y4RVNrH/D7G3Y4RVNrH.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/D7N2EKCX4Sj/D7N2EKCX4Sj.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/dhjEzFoUFzH/dhjEzFoUFzH.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/E9uDoFAP3SH/E9uDoFAP3SH.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/e9zR4mvMWw7/e9zR4mvMWw7.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/EDJbREhghzL/EDJbREhghzL.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/fzynW3qQPVF/fzynW3qQPVF.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/GdvgFV5R1Z5/GdvgFV5R1Z5.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/gTV8FGcVJC9/gTV8FGcVJC9.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/gxdoqLR6rwA/gxdoqLR6rwA.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/gYvKGZ5eRqb/gYvKGZ5eRqb.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/gZ6f7yhEvPG/gZ6f7yhEvPG.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/HxpKQynjfin/HxpKQynjfin.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/i5noydFURQK/i5noydFURQK.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/JeFG25nYj2p/JeFG25nYj2p.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/JF19kD82Mey/JF19kD82Mey.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/jh4fc5c5qoQ/jh4fc5c5qoQ.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/JmbYfDe2QKZ/JmbYfDe2QKZ.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/jtcxE69GiFV/jtcxE69GiFV.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/kEZ7cmS4wCh/kEZ7cmS4wCh.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/mJXqzFtmKg4/mJXqzFtmKg4.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/oLBMNvg9in8/oLBMNvg9in8.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/p5wJjkQkbXX/p5wJjkQkbXX.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/pa4otMbVnkk/pa4otMbVnkk.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/pLe4wQe7qrG/pLe4wQe7qrG.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/Pm6F8kyY3z2/Pm6F8kyY3z2.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/pRbA3pwrgk9/pRbA3pwrgk9.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/PuKPg4mmafe/PuKPg4mmafe.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/PX4nDJXEHrG/PX4nDJXEHrG.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/q9vSo1VnCiC/q9vSo1VnCiC.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/qoiz87JEwZ2/qoiz87JEwZ2.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/QUCTc6BB5sX/QUCTc6BB5sX.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/r1Q1Z4BcV1o/r1Q1Z4BcV1o.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/r47D5H71a5s/r47D5H71a5s.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/rPc6DW4iMge/rPc6DW4iMge.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/RPmz2sHmrrY/RPmz2sHmrrY.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/rqfALeAoiTq/rqfALeAoiTq.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/s8pcmisQ38h/s8pcmisQ38h.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/S9hNv5qa7GM/S9hNv5qa7GM.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/sKLMLpTHeUy/sKLMLpTHeUy.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/SN83YJsR3w2/SN83YJsR3w2.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/sT4fr6TAbpF/sT4fr6TAbpF.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/TbHJrupSAjP/TbHJrupSAjP.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/ULsKaCPVFJR/ULsKaCPVFJR.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/uNb9QFRL6hY/uNb9QFRL6hY.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/ur6pFq6Qu1A/ur6pFq6Qu1A.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/UwV83HsGsw3/UwV83HsGsw3.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/Uxmj2M2itWa/Uxmj2M2itWa.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/V2XKFyX4ASd/V2XKFyX4ASd.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/VFuaQ6m2Qom/VFuaQ6m2Qom.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/VLzqgDo317F/VLzqgDo317F.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/Vt2qJdWjCF2/Vt2qJdWjCF2.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/VVfe2KiqLaN/VVfe2KiqLaN.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/Vvot9Ly1tCj/Vvot9Ly1tCj.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/vyrNrziPKCB/vyrNrziPKCB.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/VzqfbhrpDEA/VzqfbhrpDEA.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/wc2JMjhGNzB/wc2JMjhGNzB.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/WYY7iVyf5p8/WYY7iVyf5p8.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/X7HyMhZNoso/X7HyMhZNoso.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/XcA2TqTSSAj/XcA2TqTSSAj.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/YFuZgdQ5vWj/YFuZgdQ5vWj.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/YmJkqBEsHnH/YmJkqBEsHnH.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/yqstnuAEVhm/yqstnuAEVhm.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/YVUC4YcDtcY/YVUC4YcDtcY.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/ZMojNkEp431/ZMojNkEp431.glb",
    "/habitat-challenge-data/data/scene_datasets/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb",
]
BATCH_SIZE = 4
# DIST = 10
DIST = 100

import collections
from typing import List, Tuple, Union

import numpy as np
from numpy import bool_, float32, float64, ndarray
from quaternion import quaternion

from habitat_sim.utils.data.pose_extractor import TopdownView

@registry.register_pose_extractor(name="panorama_extractor")
class PanoramaExtractor(PoseExtractor):
    def __init__(
        self,
        topdown_views: List[Tuple[TopdownView, str, Tuple[float32, float32, float32]]],
        meters_per_pixel: float = 0.1,
    ) -> None:
        super().__init__(topdown_views, meters_per_pixel)

    def extract_poses(
        self, view: ndarray, fp: str
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        # Determine the physical spacing between each camera position
        height, width = view.shape
        # dist = min(height, width) // 10  # We can modify this to be user-defined later
        # dist = min(height, width) // 100  # We can modify this to be user-defined later
        dist = min(height, width) // DIST  # We can modify this to be user-defined later
        dist = max(2,dist)

        # Create a grid of camera positions
        n_gridpoints_width, n_gridpoints_height = (
            width // dist - 1,
            height // dist - 1,
        )

        # Exclude camera positions at invalid positions
        gridpoints = []
        for h in range(n_gridpoints_height):
            for w in range(n_gridpoints_width):
                point = (dist + h * dist, dist + w * dist)
                if self._valid_point(*point, view):
                    gridpoints.append(point)

        # Find the closest point of the target class to each gridpoint
        poses = []
        for point in gridpoints:
            point_label_pairs = self._panorama_extraction(point, view, dist)
            poses.extend([(point, point_, fp) for point_, label in point_label_pairs])

        # Returns poses in the coordinate system of the topdown view
        return poses

    def _panorama_extraction(
        self, point: Tuple[int, int], view: ndarray, dist: int
    ) -> List[Tuple[Tuple[int, int], float]]:
        in_bounds_of_topdown_view = lambda row, col: 0 <= row < len(
            view
        ) and 0 <= col < len(view[0])
        point_label_pairs = []
        r, c = point
        neighbor_dist = dist // 2
        neighbors = [
            (r - neighbor_dist, c - neighbor_dist),
            (r - neighbor_dist, c),
            (r - neighbor_dist, c + neighbor_dist),
            (r, c - neighbor_dist),
            (r, c + neighbor_dist),
            (r + neighbor_dist, c - neighbor_dist),
            # (r + step, c), # Exclude the pose that is in the opposite direction of habitat_sim.geo.FRONT, causes the quaternion computation to mess up
            (r + neighbor_dist, c + neighbor_dist),
        ]

        for n in neighbors:
            # Only add the neighbor point if it is navigable. This prevents camera poses that
            # are just really close-up photos of some object
            if in_bounds_of_topdown_view(*n) and self._valid_point(*n, view):
                point_label_pairs.append((n, 0.0))

        return point_label_pairs


# @registry.register_pose_extractor(name="random_pose_extractor")
# class RandomPoseExtractor(PoseExtractor):
#     def extract_poses(self, view, fp):
#         height, width = view.shape
#         num_random_points = 4
#         points = []
#         while len(points) < num_random_points:
#             # Get the row and column of a random point on the topdown view
#             row, col = np.random.randint(0, height), np.random.randint(0, width)
#
#             # Convenient method in the PoseExtractor class to check if a point
#             # is navigable
#             if self._valid_point(row, col, view):
#                 points.append((row, col))
#
#         poses = []
#
#         # Now we need to define a "point of interest" which is the point the camera will
#         # look at. These two points together define a camera position and angle
#         for point in points:
#             r, c = point
#             point_of_interest = (r - 1, c) # Just look forward
#             pose = (point, point_of_interest, fp)
#             poses.append(pose)
#
#         return poses

def mask2bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, rmin, cmax - cmin, rmax - rmin

# def mask2bbox(mask):
#     rows = torch.any(mask, dim=1)
#     cols = torch.any(mask, dim=0)
#     rmin, rmax = torch.where(rows)[0][[0, -1]]
#     cmin, cmax = torch.where(cols)[0][[0, -1]]
#
#     return cmin, rmin, cmax - cmin, rmax - rmin

def show_batch(sample_batch):
    def show_row(imgs, batch_size, img_type):
        plt.figure(figsize=(12, 8))
        for i, img in enumerate(imgs):
            ax = plt.subplot(1, batch_size, i + 1)
            ax.axis("off")
            if img_type == 'rgb':
                plt.imshow(img.numpy().transpose(1, 2, 0))
            elif img_type == 'truth':
                plt.imshow(img.numpy())

        plt.show()

    batch_size = len(sample_batch['rgb'])
    for k in sample_batch.keys():
        show_row(sample_batch[k], batch_size, k)

class SemanticSegmentationDataset(Dataset):
    def __init__(self, extractor, transforms=None):
        # super(SemanticSegmentationDataset, self).__init__()

        # Define an ImageExtractor
        self.extractor = extractor

        # We will perform preprocessing transforms on the data
        self.transforms = transforms

        # Habitat sim outputs instance id's from the semantic sensor (i.e. two
        # different chairs will be marked with different id's). So we need
        # to create a mapping from these instance id to the class labels we
        # want to predict. We will use the below dictionaries to define a
        # funtion that takes the raw output of the semantic sensor and creates
        # a 2d numpy array of out class labels.
        # self.labels = {
        #     'background': 0,
        #     'wall': 1,
        #     'floor': 2,
        #     'ceiling': 3,
        #     'chair': 4,
        #     'table': 5,
        # }


        self.labels = {'': 0, 'chair': 1, 'table': 2, 'picture': 3, 'cabinet': 4, 'cushion': 5, 'sofa': 6, 'bed': 7, 'chest_of_drawers': 8, 'plant': 9, 'sink': 10, 'toilet': 11, 'stool': 12, 'towel': 13, 'tv_monitor': 14, 'shower': 15, 'bathtub': 16, 'counter': 17, 'fireplace': 18, 'gym_equipment': 19, 'seating': 20, 'clothes': 21, 'shelving': 0, 'floor': 0, 'ceiling': 0, 'stairs': 0, 'wall': 0, 'objects': 0, 'window': 0, 'blinds': 0, 'misc': 0, 'void': 0, 'mirror': 0, 'beam': 0, 'door': 0, 'lighting': 0, 'curtain': 0, 'appliances': 0, 'column': 0, 'railing': 0, 'board_panel': 0, 'furniture': 0, 'unlabeled': 0}

        self.instance_id_to_name = self.extractor.instance_id_to_name
        self.map_to_class_labels = np.vectorize(
            lambda x: self.labels.get(self.instance_id_to_name.get(x, 0), 0)
        )


    def __len__(self):
        return len(self.extractor)

    # def __getitem__(self, idx):
    #     sample = self.extractor[idx]
    #     raw_semantic_output = sample['semantic']
    #     # print("raw_semantic_output",raw_semantic_output)
    #     truth_mask = self.get_class_labels(raw_semantic_output)
    #
    #     output = {
    #         'rgb': sample['rgba'][:, :, :3],
    #         'truth': truth_mask.astype(int),
    #     }
    #
    #     if self.transforms:
    #         output['rgb'] = self.transforms(output['rgb'])
    #         output['truth'] = self.transforms(output['truth']).squeeze(0)
    #
    #     return output

    def __getitem__(self, idx):
        '''
        This special version separate the instances into a different binary sample
        '''

        sample = self.extractor[idx]
        raw_semantic_output = sample['semantic']

        ########################################################################
        u_sem = np.unique(raw_semantic_output)
        # print("u_sem",u_sem)
        oid_to_semantic_class = self.get_class_labels(u_sem)
        # print("oid_to_semantic_class",oid_to_semantic_class)
        oid_to_semantic_class_useful = np.nonzero(oid_to_semantic_class)
        # print("oid_to_semantic_class_useful",oid_to_semantic_class_useful)

        masks = []
        for i in oid_to_semantic_class_useful:
            for j in i:
                mask = np.where(raw_semantic_output == u_sem[j], 1.0, 0.0 )
                masks.append({'mask':mask,'semantic_class':oid_to_semantic_class[j] },)
        ########################################################################
        # print("masks",masks)

        truth_mask = self.get_class_labels(raw_semantic_output)

        output = {
            'rgb': sample['rgba'][:, :, :3],
            'truth': truth_mask.astype(int),
            'masks_per_instance': masks
        }

        if self.transforms:
            output['rgb'] = self.transforms(output['rgb'])
            output['truth'] = self.transforms(output['truth']).squeeze(0)

        return output

    def get_class_labels(self, raw_semantic_output):
        return self.map_to_class_labels(raw_semantic_output)

def generate_dataset(extractor, mode, path):
    dataset = SemanticSegmentationDataset(extractor)

    # print("dataset.__len__()",dataset.__len__(),flush=True)


    # Create a Dataloader to batch and shuffle our data
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    ################################################################################

    folder = path+mode+"/imgs/"
    if not os.path.exists(folder):
        os.makedirs(folder)


    annotations = []
    images = []

    image_id = 1
    ann_id   = 1

    filename=path+mode+"/"+'mp3d_2021_coco_style.json'
    if os.path.isfile(filename):
        print("updating existent file")

        with open(filename, 'r') as f:
            tmp_dict = json.load(f)

        image_id = len(tmp_dict['images'])+1
        ann_id = len(tmp_dict['annotations'])+1

    for i in range(len(dataset)):
        sample = dataset[i]

        if(sample['masks_per_instance']):
            # #######################################################################
            for j,mask in enumerate(sample['masks_per_instance']):
                rle = pycocotools.mask.encode( np.asfortranarray( np.expand_dims(mask['mask'],axis=-1).astype(np.uint8) ) )
                rle = rle[0]
                rle['counts'] = rle['counts'].decode('ascii')

                img = pycocotools.mask.decode(rle)

                annotations.append({
                    'id': int(ann_id),
                    'image_id': int(image_id),
                    'category_id': int(mask['semantic_class']),
                    'segmentation': rle,
                    'area': float(mask['mask'].sum()),
                    'bbox': [int(x) for x in mask2bbox(mask['mask'])],
                    'iscrowd': int(0)
                })

                ann_id+=1
            ############################################################################

            img_name="./imgs/"+str(image_id)+".jpg"
            img = sample['rgb']

            images.append({
                'id': int(image_id),
                'width': int(img.shape[1]),
                'height': int(img.shape[0]),
                'file_name': img_name
            })

            im = Image.fromarray(img)
            im.save(path+mode+"/"+img_name, quality=95, subsampling=0)

            image_id += 1
            ####################################################################

    info = {
        'year': int(2021),
        'version': int(1),
        'description': 'Matterport3D-Habitat-challenge-2021',
    }

    categories = [{'id': x+1} for x in range(21)]

    filename=path+mode+"/"+'mp3d_2021_coco_style.json'
    print("saving",filename)

    if os.path.isfile(filename):
        print("updating existent file")

        with open(filename, 'r') as f:
            tmp_dict = json.load(f)

        images = tmp_dict['images']+images
        annotations = tmp_dict['annotations']+annotations

        with open(filename, 'a') as f:
            json.dump({
                'info': info,
                'images': images,
                'annotations': annotations,
                'licenses': {},
                'categories': categories
            }, f)
    else:
        print("creating new file")
        with open(filename, 'w') as f:
            json.dump({
                'info': info,
                'images': images,
                'annotations': annotations,
                'licenses': {},
                'categories': categories
            }, f)
################################################################################


# starting_idx_train = 5307
# starting_idx_test = 2312
# for scene in SCENE_FILEPATHS:
for scene in SCENE_FILEPATHS[1:]:
    print("scene",scene)
    # split: A tuple of train/test split percentages. Must add to 100. Default (70, 30).
    extractor = ImageExtractor(scene_filepath=scene, img_size=(480,640), output=['rgba', 'semantic'], pose_extractor_name="panorama_extractor", split=(70,30))
    print("len(extractor)",len(extractor),flush=True)
    # mode (str): The mode to set the extractor to. Options are 'full', 'train', or 'test'.

    path = "/habitat-challenge-data/mp3d_2021_coco_style/"
    mode = "val"
    extractor.set_mode("test")
    generate_dataset(extractor, mode, path)
    mode = "train"
    extractor.set_mode("train")
    generate_dataset(extractor, mode, path)

    extractor.close()
