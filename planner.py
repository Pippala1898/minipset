import torchvision
import torch
from PIL import Image
from queue import Queue
from utils import show2, find_num_stacks
import copy
import itertools
import more_itertools
import numpy as np


class Node:
    def __init__(self, sg, ob_names, if_goal=False):
        self.sg = sg
        self.key = hash_sg(sg, ob_names)

        self.visited = False
        self.parent = None
        self.act = None
        self.goal = if_goal

    def get_key(self):
        return self.key

    def get_sg(self):
        return self.sg

    def __eq__(self, other):
        immutable_other = other.get_sg()
        immutable_other = set(tuple(rel) for rel in immutable_other)
        immutable_self = self.get_sg()
        immutable_self = set(tuple(rel) for rel in immutable_self)
        same_sg = immutable_other == immutable_self
        return same_sg


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


# For Actions
def two_stack_position(scene_image):
    tracing_two_blocks = True
    for i in range(127, -1, -1):
        indices = scene_image[i].nonzero().squeeze().cpu().numpy()
        try:
            if len(indices) > 0:
                res = [list(group) for group in more_itertools.consecutive_groups(indices)]

                if len(res) == 2:
                    if tracing_two_blocks is True:
                        tracing_two_blocks = [max(res[0]), min(res[1])]
                    else:
                        if min(res[1]) - max(res[0]) < tracing_two_blocks[1] - tracing_two_blocks[0]:
                            tracing_two_blocks = [max(res[0]), min(res[1])]
        except TypeError:
            pass
    if tracing_two_blocks[1] - tracing_two_blocks[0] > 128 / 3:
        return 0, 2
    if tracing_two_blocks[1] > 128 / 2:
        return 1, 2
    return 0, 1


def one_stack_position(scene_image):
    all_res = []
    tracing_one_blocks = True
    for i in range(127, -1, -1):
        indices = scene_image[i].nonzero().squeeze().cpu().numpy()
        try:
            if len(indices) > 0:
                res = [list(group) for group in more_itertools.consecutive_groups(indices)]

                if len(res) == 1:
                    if tracing_one_blocks is True:
                        tracing_one_blocks = np.mean(res[0])
        except TypeError:
            pass
    if tracing_one_blocks < 128 / 3:
        return 0
    if tracing_one_blocks > 2 * 128 / 3:
        return 2
    return 1


# find the position of two stacks
def blocks_positions(masks):  # which is 5*3*128*128
    scene_img = torch.sum(masks, dim=0)  # sum over 5obj
    scene_img = torch.sum(scene_img, dim=0)  # sum over 3 channels
    nb_blocks_per_img = find_num_stacks(scene_img)  # 128 * 128 , height * width
    if nb_blocks_per_img == 3:
        return None
    elif nb_blocks_per_img == 2:
        return two_stack_position(scene_img)
    return one_stack_position(scene_img)


def adding_virtual_base(sg_from, masks):  # base_objs = ['grey', 'purple', 'cyan']
    left_rel = [rel for rel in sg_from if rel[1] == "left"]
    ordered_base_obj = []

    basic_colors = ['blue', 'brown', 'green', 'pink', 'red']
    base_objs = ['grey', 'purple', 'cyan']
    # just one stack
    if len(left_rel) == 0:
        # order of that stack
        up = [ele[0] for ele in sg_from]
        down = [ele[2] for ele in sg_from]
        obj_up = [e for e in down if e not in up][0]
        obj = base_objs[blocks_positions(masks)]  # the stack would be put on here
        new_seg = copy.deepcopy(sg_from)
        new_seg.append([obj_up, 'up', obj])
        new_seg.append(['grey', 'left', 'purple'])
        new_seg.append(['purple', 'left', 'cyan'])
        return new_seg

    # two stacks
    elif len(left_rel) == 1:
        up_obj_1 = left_rel[0][0]
        up_obj_2 = left_rel[0][2]
        obj_1 = base_objs[blocks_positions(masks)[0]]
        obj_2 = base_objs[blocks_positions(masks)[1]]
        new_seg = []
        for rel in sg_from:
            if rel[1] == 'up':
                new_seg.append(rel.copy())
                if rel[2] == up_obj_1:
                    new_seg.append([up_obj_1, 'up', obj_1])
                elif rel[2] == up_obj_2:
                    new_seg.append([up_obj_2, 'up', obj_2])
        current_cover_obj = set(itertools.chain(*new_seg))

        if up_obj_1 not in current_cover_obj:
            new_seg.insert(0, [up_obj_1, 'up', obj_1])
        if up_obj_2 not in current_cover_obj:
            new_seg.append([up_obj_2, 'up', obj_2])

        new_seg.append(['grey', 'left', 'purple'])
        new_seg.append(['purple', 'left', 'cyan'])
        return new_seg

    else:  # three stacks
        left_most = left_rel[0][0]
        if left_most == left_rel[1][2]:  # [a left b], [c left a] --> c a b
            up_obj_1, up_obj_2, up_obj_3 = left_rel[1][0], left_most, left_rel[0][2]
        else:  # [a left b], [b left c] --> a b c
            up_obj_1, up_obj_2, up_obj_3 = left_most, left_rel[0][2], left_rel[1][2]
        obj_1 = 'grey'
        obj_2 = 'purple'
        obj_3 = 'cyan'
        new_seg = copy.deepcopy(sg_from)
        new_seg.append([up_obj_1, 'up', obj_1])
        new_seg.append([up_obj_2, 'up', obj_2])
        new_seg.append([up_obj_3, 'up', obj_3])
        # Remove left relations between real bases and add up relations between real and virtual bases
        for rel in left_rel:
            new_seg.remove(rel)
        # Add links between virtual bases
        new_seg.append(['grey', 'left', 'purple'])
        new_seg.append(['purple', 'left', 'cyan'])
        return new_seg


def action_model(sg_from, action):
    assert action in ["12", "13", "21", "23", "31", "32"]
    base_objs = ['grey', 'purple', 'cyan']
    relations_from = copy.deepcopy(sg_from)
    block_from = int(action[0]) - 1
    block_to = int(action[1]) - 1

    # check valid action
    up_rel = [rel for rel in relations_from if rel[1] == "up"]
    ob_to = [rel[2] for rel in up_rel]
    if base_objs[block_from] not in ob_to:
        return None

    # modify "from" block
    top_block_from, to_be_removed_rel = find_top(up_rel, base_objs[block_from])
    relations_from.remove(to_be_removed_rel)

    # modify "to" block
    top_block_to, _ = find_top(up_rel, base_objs[block_to])
    relations_from.append([top_block_from, "up", top_block_to])
    assert top_block_from != top_block_to

    return relations_from


def possible_next_states(current_state):
    predefined_actions = ["12", "13", "21", "23", "31", "32"]
    res = []
    for action in predefined_actions:
        next_state = action_model(current_state, action)
        if next_state is not None:
            res.append((next_state, action))
    return res


def hash_sg(relationships, ob_names):
    """
    hash into unique ID
    :param relationships: [['brown', 'left', 'purple'] , ['yellow', 'up', 'yellow']]
    :param ob_names:
    :return:
    """
    return tuple(set(tuple(rel) for rel in relationships))


def inv_hash_sg(a_key, ob_names):
    predefined_objects1 = ob_names[:]
    predefined_objects2 = ob_names[:]
    id2pred = {0: "none", 1: "left", 2: "up"}

    sg = []
    idx = 0
    for ob1 in predefined_objects1:
        for ob2 in predefined_objects2:
            if a_key[idx] > 0:
                sg.append([ob1, id2pred[a_key[idx]], ob2])
            idx += 1
    return sg


def find_action_plan(start_sg, end_sg, ob_names):
    # bfs
    Q = Queue()
    start_v = Node(start_sg, ob_names)
    goal_state = Node(end_sg, ob_names, if_goal=True)
    all_nodes = {start_v.get_key(): start_v, goal_state.get_key(): goal_state}
    start_v.visited = True
    Q.put(start_v)
    while not Q.empty():
        v = Q.get()
        if v == goal_state:
            goal_state = v
            break
        for w, act in possible_next_states(v.get_sg()):
            w_key = hash_sg(w, ob_names)
            if w_key not in all_nodes:
                all_nodes[w_key] = Node(w, ob_names)
            if not all_nodes[w_key].visited:
                all_nodes[w_key].visited = True
                all_nodes[w_key].parent = v
                all_nodes[w_key].act = act
                Q.put(all_nodes[w_key])
    traces = []
    actions = []
    while True:
        traces.insert(0, goal_state.get_key())
        actions.insert(0, goal_state.act)
        if goal_state == start_v:
            break
        goal_state = goal_state.parent
    return traces, actions


def visualize_plan(im_list, perrow=9, if_save=False, name="solution"):
    im_tensors = []
    transform = torchvision.transforms.ToTensor()
    for idx, im_name in enumerate(im_list):
        im = Image.open(im_name).convert('RGB')
        im_tensors.append(transform(im).unsqueeze(0))
        if if_save:
            im.save("figures/%d.png" % idx)
    im_tensors = torch.cat(im_tensors, dim=0)
    show2([im_tensors], name, perrow)


def primitive_state_visualization(trace):
    vis_string = ""
    virtual_bases = ("grey", "purple", "cyan")  # in order
    bottom_string = ""

    # Construct bottom string
    for base in virtual_bases:
        bottom_string += "{:10}".format(base)
    # vis_string = vis_string + bottom_string + "\n"  # comment out: don't print virtual bases in visualization

    # Construct layers above
    current_tops = list(virtual_bases)
    while True:
        layer_list = ["", "", ""]
        new_tops = ["", "", ""]
        layer_string = ""
        for relation in trace:
            if relation[1] == "up" and relation[2] in current_tops:
                stack_index = current_tops.index(relation[2])
                layer_list[stack_index] = relation[0]
                new_tops[stack_index] = relation[0]
        if len("".join(layer_list)) == 0:  # layer_list is still ["", "", ""], meaning no objects at this layer
            break  # done
        for obj in layer_list:
            layer_string += "{:10}".format(obj)
        vis_string = layer_string + "\n" + vis_string  # add to string as layer above
        current_tops = new_tops
    return vis_string


def primitive_plan_visualization(traces, actions):
    """Given the traces (first output of find_action_plan()), nicely print out intermediate configurations of the action
     plan"""
    for trace, action in zip(traces, actions):
        configuration = primitive_state_visualization(trace)
        if action is None:
            print("Initial State")
        else:
            print(f"Move block at top of stack {action[0]} to top of stack {action[1]}")
        print(configuration)
    return
