import json
import torch
import torchvision
import numpy as np
import pickle
import random
from utils import read_seg_masks
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from os import listdir
from os.path import isfile, join


class SimData(Dataset):
    def __init__(self, root_dir="data/sim_data/5objs_seg", nb_samples=10,
                 train=True, train_size=0.6, save_data=True):
        print("Loading from", root_dir)
        super(SimData, self).__init__()
        self.root_dir = root_dir
        self.train = train
        identifier = root_dir.split("/")[-1]

        self.scene_jsons = [join(root_dir, f) for f in listdir(root_dir) if isfile(join(root_dir, f))]
        print("there are %d files total" % len(self.scene_jsons))
        if nb_samples > 0:
            random.shuffle(self.scene_jsons)
            self.scene_jsons = self.scene_jsons[:nb_samples]

        name = "json2sg-%s-%d" % (identifier, nb_samples)
        if isfile("data/%s" % name):
            print("Loading precomputed json2sg:", "data/%s" % name)
            with open("data/%s" % name, 'rb') as f:
                self.js2data = pickle.load(f)
        else:
            self.js2data = {}
            for js in self.scene_jsons:
                self.js2data[js] = read_seg_masks(js)
            if save_data:
                with open("data/%s" % name, 'wb') as f:
                    pickle.dump(self.js2data, f)

        keys = list(self.js2data.keys())
        if train:
            self.data = {du3: self.js2data[du3] for du3 in keys[:int(len(keys)*train_size)]}
        else:
            self.data = {du3: self.js2data[du3] for du3 in keys[int(len(keys)*train_size):]}
        self.keys = list(self.data.keys())
        print("loaded", len(self.js2data), "used", len(self.data))

    def clear(self):
        self.js2data.clear()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        return self.data[self.keys[item]]


def assemble_scene_graph(obj_names, locations, return_assigns=False, if_add_bases=True):
    """
    reconstruct a sg from object names and coordinates
    """
    location_dict = {}
    objects = []

    if type(locations) == torch.Tensor:
        locations = locations.cpu().numpy()
    elif isinstance(locations, list):
        locations = np.array(locations)

    locations = locations.reshape(-1, 2)
    k_means_assign = kmeans(locations[:, 0])

    for idx, object_id in enumerate(obj_names):
        a_key = k_means_assign[idx]
        if a_key not in location_dict:
            location_dict[a_key] = [(object_id, locations[idx][1])]
        else:
            location_dict[a_key].append((object_id, locations[idx][1]))
        objects.append(object_id)
    relationships = []
    if if_add_bases:
        relationships.extend([
            ["brown", "left", "purple"],
            ["purple", "left", "cyan"],
        ])
    for du3 in location_dict:
        location = sorted(location_dict[du3], key=lambda x: x[1])
        while len(location) > 1:
            o1 = location.pop()[0]
            o2 = location[-1][0]
            relationships.append([o1, "up", o2])
    if return_assigns:
        return relationships, k_means_assign
    return relationships


def sim_collate_fn(batch):
    all_imgs, all_imgs2, weights = [], [], []
    sgs = []
    names = []
    im_names = []
    for masks, def_mat, def_wei, ob_names, sg, im_name in batch:
        all_imgs.append(masks.unsqueeze(0))
        all_imgs2.append(def_mat.unsqueeze(0))
        weights.append(def_wei.unsqueeze(0))
        sgs.append(sg)
        names.append(ob_names)
        im_names.append(im_name)

    all_imgs = torch.cat(all_imgs)
    all_imgs2 = torch.cat(all_imgs2)
    weights = torch.cat(weights)

    return all_imgs, all_imgs2, weights, sgs, names, im_names


def evaluation(json2im, model, loss_func, device="cuda"):
    val_loss = []
    correct = 0
    total = 0.0
    best_p = None
    best_l = None
    best_a = None
    print("evaluating %d samples" % len(json2im))

    all_images = []
    for val_json in json2im:
        val_batch = json2im[val_json][:2]
        images, coords = [tensor.to(device).double() for tensor in val_batch]
        all_images.append(images)
    all_images = torch.cat(all_images, dim=0)
    im_iter = DataLoader(all_images, batch_size=64)
    all_pred_coords = []
    for im in im_iter:
        with torch.no_grad():
            pred_coords = model(im)
            all_pred_coords.append(pred_coords)
    all_pred_coords = torch.cat(all_pred_coords, dim=0)

    for idx, val_json in enumerate(json2im):
        val_batch = json2im[val_json][:2]
        obj_names = json2im[val_json][2]
        images, coords = [tensor.to(device).double() for tensor in val_batch]
        sg = assemble_scene_graph(obj_names, coords)

        pred_coords = all_pred_coords[idx]
        pred_sg, assigns = assemble_scene_graph(obj_names, pred_coords, return_assigns=True)
        correct += sg == pred_sg
        total += 1
        loss = loss_func(pred_coords, coords)
        val_loss.append(loss.item())
        if best_l is None or loss.item() < best_l:
            best_l = loss.item()
            best_p = (pred_coords, coords)
            best_a = assigns
    print("acquire this best:")
    print("assigns", best_a)
    print("pred\n", best_p[0])
    print("true\n", best_p[1])
    print()
    acc = correct / total
    return np.mean(val_loss), acc


def find_top(up_rel, ob_start):
    rel_res = None
    while True:
        done = True
        for rel in up_rel:
            if rel[2] == ob_start:
                ob_start = rel[0]
                done = False
                rel_res = rel
                break
        if done:
            return ob_start, rel_res


def kmeans(data_):
    c1 = max(data_)
    c2 = min(data_)
    c3 = (c1+c2)/2
    c_list = [c1, c2, c3]
    assign = [0]*len(data_)
    for _ in range(10):
        for idx, d in enumerate(data_):
            assign[idx] = min([0, 1, 2], key=lambda x: (c_list[x]-d)**2)

        for c in range(3):
            stuff = [d for idx, d in enumerate(data_) if assign[idx] == c]
            if len(stuff) > 0:
                c_list[c] = sum(stuff)/len(stuff)
    return assign
