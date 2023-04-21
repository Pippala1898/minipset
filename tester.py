import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import cv2
from utils import visualize_initial_and_goal_states, read_seg_masks, \
    find_num_stacks
from planner import adding_virtual_base, find_action_plan, primitive_plan_visualization

def get_and_vis_state_masks(image_path, visualize=True):
    '''
    given a path, the function would return the state masks, defualt masks and object names. 
    The object names are the names (colors) of objects contained in masks.
    '''
    state_masks, default_masks, _, obj_names, _, _ = read_seg_masks(image_path)  
    img = Image.open(image_path)
    img_array = np.array(img)
    img_tensor = torch.from_numpy(img_array)    
    if visualize:
        plt.figure()
        plt.imshow(img_tensor)
        plt.axis("off")
        plt.title('Image')
    return state_masks, default_masks, obj_names

path = "data/sim_data/minipset_data_1a/z.seg0_s0_01234_s1__s2_.ppm"
state_masks, default_masks, obj_names = get_and_vis_state_masks(path, False)

def test_masks(reconstruct_image):
    path = "data/sim_data/minipset_data_1a/z.seg0_s0_01234_s1__s2_.ppm"
    state_masks, default_masks, obj_names = get_and_vis_state_masks(path, False)
    img = Image.open(path)
    imge_array = np.array(img)
    # Load  images
    img1 = imge_array
    img2 = torch.permute(reconstruct_image(state_masks), (1,2,0)).numpy()
    # Threshold the images to create binary masks
    threshold = 0   
    binary1 = (img1 > threshold ).astype(float)
    binary2 = (img2 > threshold ).astype(float)

    # Calculate the intersection and union of the binary masks
    intersection = np.logical_and(binary1, binary2).astype(float)
    union = np.logical_or(binary1, binary2).astype(float)

    # Calculate the area of the intersection and union
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)

    # Calculate the IoU
    iou = intersection_area / union_area
    if iou<0.999:
         raise Exception('The reconstructed image is the same as the original image.') 
    print('Passed!!!') 
         



    
def test_initial_scene_graph(scene_graph):
    if len(scene_graph) != 4:
        raise Exception('The number of relations is Wrong') 
    if set(ele[1] for ele in scene_graph) != {'up'}:
        raise Exception('The relations are Wrong')    
    for ele in scene_graph:
        if ele[0] =='pink' and ele[1] !='navy':
            raise Exception('The relations of the pink object are Wrong')   
        if ele[0] =='navy' and ele[1] !='brown':
            raise Exception('The relations of the navy object are Wrong')  
        if ele[0] =='brown' and ele[1] !='green':
            raise Exception('The relations of the brown object are Wrong')   
        if ele[0] =='green' and ele[1] !='red':
            raise Exception('The relations of the green object are Wrong')          
    print('Passed!!!')      


data1, stacks1 = torch.tensor([0.9103, 0.2792, 0.2227, 0.2042, 1.4573]) , 3
answer1 = [1, 0, 0, 0, 2]
data2, stacks2 = torch.tensor([0.8027, 1.4846, 0.2536, 0.1898, 0.2269, 0.8657]), 3
answer2 = [1, 2, 0, 0, 0, 1]
def test_kmeans(fn):
    if fn(data1, stacks1) != answer1:
         raise Exception('The K-means function is not correct.') 
    if fn(data2, stacks2) != answer2:
        raise Exception('The K-means function is not correct.')  
    print('Passed!!!')      



obj_names1, coordinates1, nb_clusters1 = ['blue', 'pink', 'brown', 'green', 'red', 'navy']  , torch.tensor([[[0.8027, 0.1444],
         [1.4846, 0.2351],
         [0.2536, 0.5668],
         [0.1898, 0.4035],
         [0.2269, 0.2334],
         [0.8657, 0.3258]]]) , 3

obj_names2, coordinates2, nb_clusters2 =  ['pink', 'brown', 'green', 'red', 'navy'] , torch.tensor([[[0.9103, 0.2694],
         [0.2792, 0.3699],
         [0.2227, 0.1842],
         [0.2042, 0.5347],
         [1.4573, 0.1924]]]) , 3


def test_SG(fn):
    if fn(obj_names1, coordinates1,nb_clusters1) != [['navy', 'up', 'blue'],
 ['brown', 'up', 'green'],
 ['green', 'up', 'red'],
 ['red', 'left', 'blue'],
 ['blue', 'left', 'pink']]:
        raise Exception('The relationship function is not correct.') 
    if fn(obj_names2, coordinates2, nb_clusters2) != [['red', 'up', 'brown'],
 ['brown', 'up', 'green'],
 ['green', 'left', 'pink'],
 ['pink', 'left', 'navy']]:
       raise Exception('The relationship function is not correct.')  
    print('Passed!!!')





