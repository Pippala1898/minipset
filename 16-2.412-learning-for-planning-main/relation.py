#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 07:31:27 2023

@author: pippala
"""


import sys
import pickle
from PIL import Image
from queue import Queue
from models import LocationBasedGenerator
from utils import show2, read_seg_masks, name2sg, read_seg_masks_slow, find_num_stacks, read_seg_masks_slow, assemble_scene_graph 
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from models import LocationBasedGenerator
import numpy as np
import torch
import argparse
import datetime
import planner
from planner import adding_base, find_action_plan, primitive_plan_visualization



model = LocationBasedGenerator()
model.to('cpu')

weights = torch.load('pre_models/model-sim-20230322-213256-5objs_seg')
model.load_state_dict(weights)

# Start From Here
im_dir ="data/sim_data/5objs_seg/z.seg1215_s0_120_s1_4_s2_3.ppm"
dir_12 = im_dir
masks, def_mat, wei_mat, ob_names, relation  = read_seg_masks_slow(im_dir)


# Ends Here
dir_1 = 'data/sim_data/5objs_seg/z.seg1279_s0_032_s1_4_s2_1.ppm'
masks_out, def_mat_out, wei_mat_out, ob_names_out, relation_out = read_seg_masks_slow(dir_1)

# Other examples
#
# dir_13 = 'data/sim_data/5objs_seg/z.seg1226_s0_013_s1__s2_42.ppm'
# m, d, w, o, r = read_seg_masks_slow(dir_13)
#
# dir_23 = 'data/sim_data/5objs_seg/z.seg1970_s2_014_s0__s1_32.ppm'
# m_, d_, w_, o_, r_ = read_seg_masks_slow(dir_23)


# First of all, let me show how u and u-def look like

# # Display the image using matplotlib
# for i in range(5):
#     img_array = masks[i].permute((1,2,0)).numpy()
#     plt.imshow(img_array)
#     plt.savefig('/Users/pippala/Desktop/psetcode/img/mask'+str(i)+'.png')
#
#
#
# for i in range(5):
#     img_array = def_mat[i].permute((1,2,0)).numpy()
#     plt.imshow(img_array)
#     plt.savefig('/Users/pippala/Desktop/psetcode/img/default'+str(i)+'.png')
#
    
# Now Given an image (dir) we can ask to calculate masks (u) and u-def 
# Correct Answer: using read_seg_masks_slow() from utils file

# Second Calculate the coordinates PHI, which is too tough to be a problem

# coordinates/ locations

x = masks.unsqueeze(0)
x = x.view(-1, 3, 128, 128)
batch_size = 1
nb_objects = 5
alpha = model.find_alpha(x).view(batch_size, nb_objects, 6).detach().numpy() # five objects, length of coordinates for each object is 2
locations = alpha[:, :, [2, 5]]
locations[:, :, 0] *= -1
locations = locations[0]
# # Based on this two functions: 
    
# def return_sg(self, x, ob_names, if_return_trans_vec=False):
#     """
#     x: masks, (bs, nb masks, 3, 128, 128)
#     ob_names: name for each mask
#     """
#     scene_img = torch.sum(x, dim=1)
#     scene_img = torch.sum(scene_img, dim=1)
#     nb_blocks_per_img = [find_num_stacks(scene_img[du1]) for du1 in range(scene_img.size(0))]
#     nb_objects = x.size(1)
#     batch_size = x.size(0)
#     x = x.view(-1, 3, 128, 128)
#     with torch.no_grad():
#         alpha = self.find_alpha(x).view(batch_size, nb_objects, 6)
#     trans_vec = alpha[:, :, [2, 5]]
#     trans_vec[:, :, 0] *= -1
#     res = []

#     for i in range(batch_size):
#         sg = assemble_scene_graph(ob_names[i], trans_vec[i], nb_blocks_per_img[i])
#         res.append(sorted(sg))

#     if if_return_trans_vec:
#         return res, trans_vec

#     return res



    
# def find_alpha(self, x):  # TODO: fix notation to match the paper
#     """The matrix in Appendix A.2. Called alpha here, but alpha in the paper"""
    
#     xs = self.main(x)  # pretrained resnet / transfer learned
#     if xs.size(0) == 1:
#         xs = xs.squeeze().unsqueeze(0)
#     else:
#         xs = xs.squeeze()
#     trans_vec = self.final_layer(xs)  # custom final layer, output represents the position from the bottom left corner
#     trans_vec = torch.sigmoid(trans_vec)  # this is the a and b in the alpha matrix in the paper
#     alpha = torch.tensor([1, 0, -1.6, 0, 1, 1.6], dtype=torch.float, requires_grad=True).repeat(x.size(0), 1).to(
#         x.device)
#     alpha[:, 2] = -trans_vec[:, 0] * 1.6  # -1.6 a
#     alpha[:, 5] = trans_vec[:, 1] * 1.6  # 1.6 b
#     return alpha.view(-1, 2, 3)


# Third: calculating the number of clusters/blocks
# A: using eyes, counting the number of clusters
# B: 
scene_img = torch.sum(masks, dim=0) # sum over 5obj
scene_img = torch.sum(scene_img, dim=0) # sum over 3 channels
nb_clusters = find_num_stacks(scene_img) # 128 * 128 , height * width


# Fourth, we calculate the Scene Graph, a set of tuples {o_1, e_, o_2}, e \in {'left', 'up'}
relation = name2sg('z.seg1215_s0_120_s1_4_s2_3.ppm') # it is the SG for the original graph
# we can use the RHO function
obj_names = ob_names
relationships = assemble_scene_graph(obj_names, locations, nb_clusters, return_assigns=False) #RHO for decoding
## obj_names:= names of objects; nb_clusters = #of blocks/clusters



# Fifth - Actions:
basic_colors = ['blue','brown', 'green', 'pink', 'red']
base_objs = ['grey', 'purple', 'cyan']
start_sg = adding_virtual_base(relation, masks)       # initial state
end_sg = adding_virtual_base(relation_out, masks_out) # end state
names = basic_colors + base_objs
traces, actions = find_action_plan(start_sg, end_sg, ob_names)
print(actions)
primitive_plan_visualization(traces, actions)

# We can ask to create SG
