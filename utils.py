import logging
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import more_itertools
from torchvision.utils import make_grid
from PIL import Image





def construct_weight_map(bbox):
    weight_map = torch.ones(128, 128)
    a_ = 255
    b_ = 1
    bbox_int = [int(dm11) for dm11 in bbox]

    weight_map[:, range(0, bbox_int[0]+1)] *= torch.tensor([(b_-a_)*x_/bbox[0]+a_ for x_ in range(0, bbox_int[0]+1)])
    weight_map[:, range(bbox_int[2], 128)] *= torch.tensor([(a_-b_)*(x_-bbox[2])/(128-bbox[2])+b_ for x_ in range(bbox_int[2], 128)])

    for x_ in range(0, bbox_int[1]+1):
        weight_map[x_, :] *= (b_-a_)*x_/bbox[1]+a_
    for x_ in range(bbox_int[3], 128):
        weight_map[x_, :] *= (a_-b_)*(x_-bbox[3])/(128-bbox[3])+b_

    for x_ in range(bbox_int[0], bbox_int[2] + 1):
        for y_ in range(bbox_int[1], bbox_int[3] + 1):
            weight_map[y_, x_] = 0

    weight_map = torch.sqrt(weight_map)
    return weight_map.unsqueeze(0)


def show2(im_, name, nrow):
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)

    fig_ = plt.figure(figsize=(15, 15))
    for du3 in range(1, len(im_)+1):
        plt.subplot(1, len(im_), du3)
        plt.axis("off")
        plt.imshow(np.transpose(make_grid(im_[du3-1], padding=5, normalize=False, pad_value=50, nrow=nrow),
                                (1, 2, 0)))

    plt.axis("off")
    # plt.title("black: no action, red: 1-3, yellow: 3-1, green: 1-2, blue: 2-3, pink: 3-2, brown: 2-1")
    plt.savefig(name, transparent=True, bbox_inches='tight')
    print("saved to", name)
    plt.close(fig_)
    logger.setLevel(old_level)



def find_num_stacks(scene_image):
    all_res = []
    for i in range(127, -1, -1):
        indices = scene_image[i].nonzero().squeeze().cpu().numpy()
        try:
            if len(indices) > 0:
                res = [list(group) for group in more_itertools.consecutive_groups(indices)]
                if len(res) == 3:
                    return 3
                else:
                    all_res.append(len(res))
        except TypeError:
            pass
    return max(all_res)







def name2sg(name):
    name = name.replace(".ppm", "")
    numb2color = {
        0: "red",
        1: "green",
        2: "brown",
        3: "blue",
        4: "pink",
        5: "navy",
        6: "pink2",
    }
    infor = name.split("_")[1:]
    bottoms = [0, 0, 0]
    relationships = []

    for dm in range(len(infor)):
        if "s" in infor[dm]:
            stack = infor[dm+1]
            if len(stack) == 0:
                continue
            bottoms[int(infor[dm][1])] = numb2color[int(stack[0])]
            for dm2 in range(len(stack)-1):
                o1 = stack[dm2]
                o2 = stack[dm2+1]
                relationships.append([numb2color[int(o2)], "up", numb2color[int(o1)]])
    if bottoms[2] != 0 and bottoms[1] != 0:
        relationships.append([bottoms[1], "left", bottoms[2]])
    if bottoms[0] != 0 and bottoms[1] != 0:
        relationships.append([bottoms[0], "left", bottoms[1]])
    if bottoms[0] != 0 and bottoms[2] != 0 and bottoms[1] == 0:
        relationships.append([bottoms[0], "left", bottoms[2]])
    return sorted(relationships)


def return_default_mat(im_tensor):
    im_np = im_tensor.numpy()[0, :, :]
    a = np.where(im_np != 0)
    bbox_int = np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])

    default_inp = torch.zeros_like(im_tensor)
    idc1 = range(bbox_int[0], bbox_int[2] + 1)
    idc2 = range(len(idc1))
    for j_, y_ in enumerate(range(bbox_int[1], bbox_int[3] + 1)):
        default_inp[:, j_ + 128 - (-bbox_int[1] + bbox_int[3] + 1), idc2] = im_tensor[:, y_, idc1]

    weight = construct_weight_map(bbox_int)

    return default_inp, weight


def read_seg_masks_slow(im_dir):
    im_pil = Image.open(im_dir).convert('RGB')
    transform = torchvision.transforms.ToTensor()
    im_mat = transform(im_pil)
    cl2name = {
        (66, 0, 192): 'blue',
        (194, 0, 192): 'pink',
        (194, 128, 64): 'brown',
        (66, 128, 64): 'green',
        (194, 0, 64): 'red',
        (64, 128, 194): 'navy'
    }
    ob_names = ['blue', 'pink', 'brown', 'green', 'red', 'navy']
    name2mask = {dm3: torch.zeros(3, 128, 128) for dm3 in ob_names}

    navy_existed = False
    for i in range(im_mat.size(1)):
        for j in range(im_mat.size(1)):
            if torch.sum(im_mat[:, i, j]).item() == 0:
                continue
            color = tuple([
                int(im_mat[0, i, j].item()*255),
                int(im_mat[1, i, j].item()*255),
                int(im_mat[2, i, j].item()*255),
            ])
            if not navy_existed:
                if cl2name[color] == "navy":
                    navy_existed = True
            name2mask[cl2name[color]][:, i, j] = im_mat[:, i, j]

    if not navy_existed:
        ob_names.remove("navy")
        del name2mask["navy"]
    masks = torch.cat([name2mask[dm4].unsqueeze(0) for dm4 in ob_names], dim=0)
    def_wei = [return_default_mat(name2mask[dm4]) for dm4 in ob_names]
    def_mat = torch.cat([dm4[0].unsqueeze(0) for dm4 in def_wei], dim=0)
    wei_mat = torch.cat([dm4[1].unsqueeze(0) for dm4 in def_wei], dim=0)
    # show2([masks, def_mat, wei_mat], "masks_test", 5)

    return masks, def_mat, wei_mat, ob_names, name2sg(im_dir.split("/")[-1])


def read_seg_masks(im_dir="data/sim_data/7objs_7k/z.seg346_s1_3420615_s2__s0_.ppm"):
    im_pil = Image.open(im_dir).convert('RGB')
    transform = torchvision.transforms.ToTensor()
    im_mat = transform(im_pil)
    cl2name = {
        torch.tensor([66, 0, 192]): 'navy',
        torch.tensor([194, 0, 192]): 'pink',
        torch.tensor([194, 128, 64]): 'brown',
        torch.tensor([66, 128, 64]): 'green',
        torch.tensor([194, 0, 64]): 'red',
        torch.tensor([66, 128, 192]): 'blue',
        torch.tensor([194, 128, 192]): 'pink2',

    }
    ob_names = ['blue', 'pink', 'brown', 'green', 'red', 'navy', 'pink2']
    name2mask = {}

    blue_existed = False
    pink2_existed = False

    im_mat = im_mat*255
    im_mat = im_mat.long()

    for color in cl2name:
        selected = im_mat[0, :, :]==color[0]
        selected *= im_mat[1, :, :]==color[1]
        selected *= im_mat[2, :, :]==color[2]

        mask = im_mat * selected
        mask = mask.float()/255.0
        name2mask[cl2name[color]] = mask
        if not blue_existed:
            if torch.sum(mask).item() != 0 and cl2name[color] == "blue":
                blue_existed = True
        if not pink2_existed:
            if torch.sum(mask).item() != 0 and cl2name[color] == "pink2":
                pink2_existed = True

    if not blue_existed:
        ob_names.remove("blue")
        del name2mask["blue"]
    if not pink2_existed:
        ob_names.remove("pink2")
        del name2mask["pink2"]
    masks = torch.cat([name2mask[dm4].unsqueeze(0) for dm4 in ob_names], dim=0)
    def_wei = [return_default_mat(name2mask[dm4]) for dm4 in ob_names]
    def_mat = torch.cat([dm4[0].unsqueeze(0) for dm4 in def_wei], dim=0)
    wei_mat = torch.cat([dm4[1].unsqueeze(0) for dm4 in def_wei], dim=0)
    # show2([masks[:3], def_mat[:3]], "masks_test", 5)

    return masks, def_mat, wei_mat, ob_names, name2sg(im_dir.split("/")[-1]), im_dir.split("/")[-1]


def compute_iou(pred, true):
    nb_objects = pred.size(1)
    pred = torch.sum(pred.view(-1, 3, 128, 128), dim=1)
    true = torch.sum(true.view(-1, 3, 128, 128), dim=1)
    pred[pred.nonzero(as_tuple=True)] = 128
    true[true.nonzero(as_tuple=True)] = 128
    total = pred+true
    res = []
    for i in range(total.size(0)):
        intersect = torch.sum(total[i].flatten()==256).item()
        union = torch.sum(total[i].flatten()==128).item()
        res.append(intersect/(intersect+union)*1.0)
    compressed_res = []
    for j in range(0, len(res), nb_objects):
        compressed_res.append(res[j: j+nb_objects])
    assert np.mean(compressed_res) - np.mean(res) <= 0.00001, "%f %f" % (np.mean(compressed_res), np.mean(res))
    return compressed_res, np.mean(compressed_res)


def visualize_initial_and_goal_states(initial_image_path, goal_image_path):
    initial_state_masks, _, _, _, _, _ = read_seg_masks(initial_image_path)
    goal_state_masks, _, _, _, _, _ = read_seg_masks(goal_image_path)

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    img_array = torch.sum(initial_state_masks, dim=0).permute(1, 2, 0)
    plt.imshow(img_array)
    plt.axis("off")
    plt.title("Initial State")

    plt.subplot(1, 2, 2)
    img_array = torch.sum(goal_state_masks, dim=0).permute(1, 2, 0)
    plt.imshow(img_array)
    plt.axis("off")
    plt.title("Goal State")

    plt.show()


if __name__ == '__main__':
    import time

    for _ in range(1):
        start = time.time()
        read_seg_masks()
        end = time.time()
        print(end-start)
