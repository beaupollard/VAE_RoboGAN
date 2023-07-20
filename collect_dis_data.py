'''
For generating the Reward Net training data
example:
python3 collect_dis_data.py --data_dir new_train_data_loc_prune --save_dir 0to1k_data --task 0
'''
from sklearn.manifold import TSNE
from pickletools import read_uint1
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os

import math, sys
import numpy as np
import argparse
from fast_jtnn import *

import torch
import torch.nn as nn
# from gan_utils import get_params

from robot_utils import *
import robot_utils.tasks as tasks
import pyrobotdesign as rd
from design_search import make_graph, build_normalized_robot
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

# params = get_params()
# env_size = params["env_vect_size"]

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)

parser.add_argument('--start_index', type=int, default=0)

parser.add_argument('--collecting_size', type=int, default=1000)

parser.add_argument('--task', type=int, default=3)

parser.add_argument("-j", "--jobs", type=int, default=8,
                        help="Number of jobs/threads")
parser.add_argument("-l", "--episode_len", type=int, default=128,
                        help="Length of episode")
# --save_dir sum_ls28_pred20 --data_dir new_train_data_loc_merge --gamma 20
args = parser.parse_args()

print(args)
def sample_graph(model):
    root, pred_nodes = model.sample_prior()
    n_nodes = len(pred_nodes)
    adj_matrix_np = np.zeros([n_nodes, n_nodes])
    features_np = np.zeros(n_nodes)
    idx_offset = root.idx
    for i in range(n_nodes):
        node = pred_nodes[i]
        features_np[i] = node.wid
        for nei in node.neighbors:
            true_idx = nei.idx - idx_offset
            adj_matrix_np[true_idx, i] = 1
            adj_matrix_np[i, true_idx] = 1
    return adj_matrix_np, features_np,

def decode_graph(model, tree_vec):
    root, pred_nodes = model.decode(tree_vec, prob_decode=False)
    n_nodes = len(pred_nodes)
    adj_matrix_np = np.zeros([n_nodes, n_nodes])
    features_np = np.zeros(n_nodes)
    idx_offset = root.idx
    for i in range(n_nodes):
        node = pred_nodes[i]
        features_np[i] = node.wid
        for nei in node.neighbors:
            true_idx = nei.idx - idx_offset
            adj_matrix_np[true_idx, i] = 1
            adj_matrix_np[i, true_idx] = 1
    return adj_matrix_np, features_np


def get_robot_image(robot, task):
    sim = rd.BulletSimulation(task.time_step)
    task.add_terrain(sim)
    viewer = rd.GLFWViewer()
    if robot is not None:
        robot_init_pos, _ = presimulate(robot)
        # Rotate 180 degrees around the y axis, so the base points to the right
        sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
        robot_idx = sim.find_robot_index(robot)

        # Get robot position and bounds
        base_tf = np.zeros((4, 4), order='f')
        lower = np.zeros(3)
        upper = np.zeros(3)
        sim.get_link_transform(robot_idx, 0, base_tf)
        sim.get_robot_world_aabb(robot_idx, lower, upper)
        viewer.camera_params.position = base_tf[:3,3]
        viewer.camera_params.yaw = - np.pi / 3
        viewer.camera_params.pitch = -np.pi / 4.5
        viewer.camera_params.distance = np.linalg.norm(upper - lower) * 1.5
    else:
        viewer.camera_params.position = [1.0, 0.0, 0.0]
        viewer.camera_params.yaw = -np.pi / 3
        viewer.camera_params.pitch = -np.pi / 6
        viewer.camera_params.distance = 5.0

    viewer.update(task.time_step)
    return viewer.render_array(sim)

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

def get_dof_count(robot, task, opt_seed, thread_count, episode_count=1):
    # robot_init_pos, has_self_collision = presimulate(robot)

    # if has_self_collision:
    #     return 0 # set it to be the worst performing designs
    lower = np.zeros(3)
    upper = np.zeros(3)
    robot_init_pos = [-upper[0], -lower[1], 0.0]
    def make_sim_fn():
        sim = rd.BulletSimulation(task.time_step)
        task.add_terrain(sim)
        # Rotate 180 degrees around the y axis, so the base points to the right
        sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
        return sim

    main_sim = make_sim_fn()
    robot_idx = main_sim.find_robot_index(robot)

    dof_count = main_sim.get_robot_dof_count(robot_idx)
    
    return dof_count

def get_robot_state(sim, robot_id):
    base_tf = np.zeros((4, 4), order = 'f')
    sim.get_link_transform(robot_id, 0, base_tf)
    base_R = deepcopy(base_tf[0:3, 0:3])
    base_pos = deepcopy(base_tf[0:3, 3])

    # anguler velocity first and linear velocity next
    base_vel = np.zeros(6, order = 'f')
    sim.get_link_velocity(robot_id, 0, base_vel)
    
    n_dofs = sim.get_robot_dof_count(robot_id)
    
    joint_pos = np.zeros(n_dofs, order = 'f')
    sim.get_joint_positions(robot_id, joint_pos)
    
    joint_vel = np.zeros(n_dofs, order = 'f')
    sim.get_joint_velocities(robot_id, joint_vel)
    
    state = np.hstack((base_R.flatten(), base_pos, base_vel, joint_pos, joint_vel))

    return state

def get_distance(robot, task):
        robot_init_pos, has_self_collision = presimulate(robot)
        state_list = []
        # finding distance start from here
        
        def make_sim_fn():
            sim = rd.BulletSimulation(task.time_step)
            task.add_terrain(sim)
            # Rotate 180 degrees around the y axis, so the base points to the right
            sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
            return sim

        main_sim = make_sim_fn()
        robot_idx = main_sim.find_robot_index(robot)

        dof_count = main_sim.get_robot_dof_count(robot_idx)

        if has_self_collision:
            return 0, None, None, np.zeros((dof_count, task.episode_len)), task, dof_count, np.zeros((task.episode_len, 1))

        episode_count = 1
        if episode_count >= 2:
            value_estimator = rd.FCValueEstimator(main_sim, robot_idx, 'cpu', 64, 3, 1)
        else:
            value_estimator = rd.NullValueEstimator()
        input_sampler = rd.DefaultInputSampler()
        objective_fn = task.get_objective_fn()

        replay_obs = np.zeros((value_estimator.get_observation_size(), 0))
        replay_returns = np.zeros(0)

        for episode_idx in range(episode_count):
            optimizer = rd.MPPIOptimizer(1.0, task.discount_factor, dof_count,
                                        task.interval, task.horizon, 512,
                                        12, opt_seed + episode_idx,
                                        make_sim_fn, objective_fn, value_estimator,
                                        input_sampler)

            optimizer.update()
            optimizer.set_sample_count(64)

            main_sim.save_state()
            input_sequence = np.zeros((dof_count, task.episode_len))
            obs = np.zeros((value_estimator.get_observation_size(),
                            task.episode_len + 1), order='f')
            rewards = np.zeros(task.episode_len * task.interval)

            lower = np.zeros(3)
            upper = np.zeros(3)
            for j in range(task.episode_len):
                optimizer.update()
                input_sequence[:, j] = optimizer.input_sequence[:, 0]
                optimizer.advance(1)
                # print(type(main_sim), type(obs[:,j]), obs[:, j].shape)
                # value_estimator.get_observation(main_sim, obs[:, j])
                value_estimator.get_observation(main_sim, np.expand_dims(obs[:, j], axis=1))
                if j == 0:
                    main_sim.get_robot_world_aabb(robot_idx, lower, upper)
                    init_lower = lower.copy()
                    init_upper = upper.copy()
                    # print(init_lower, init_upper)
                
                for k in range(task.interval):
                    main_sim.set_joint_targets(robot_idx,
                                            input_sequence[:, j].reshape(-1, 1))
                    task.add_noise(main_sim, j * task.interval + k)
                    main_sim.step()
                    rewards[j * task.interval + k] = objective_fn(main_sim)
                
                if (j == task.episode_len -1):
                    main_sim.get_robot_world_aabb(robot_idx, lower, upper)
                    # print(lower, upper)
                    # print("diff", lower-init_lower, upper-init_upper)
                    # print("Center diff", (lower+upper)/2 - (init_lower+init_upper)/2)
                    dis = ((lower+upper)/2 - (init_lower+init_upper)/2)[0]
                
                cur_state = get_robot_state(main_sim, robot_idx)
                state_list.append(cur_state)
            # value_estimator.get_observation(main_sim, obs[:, -1])
            value_estimator.get_observation(main_sim, np.expand_dims(obs[:, -1], axis=1))

            main_sim.restore_state()
        print("The dis is",dis)
        # camera_params, record_step_indices = view_trajectory(main_sim, robot_idx, input_sequence, task)
        state_np = np.vstack(state_list)
        return dis, main_sim, robot_idx, input_sequence, task, dof_count, state_np

# num_of_data = 1000
# model = JTNNVAE(args.hidden_size, args.latent_size, args.depthT, args.encode, args.pred)
# model.load_state_dict(torch.load(args.model))
# model = model.cuda()

opt_seed = 42

task_class = getattr(tasks, "FlatTerrainTask")
FlatTerrainTask = task_class(episode_len=256)
task_class = getattr(tasks, "RidgedTerrainTask")
RidgedTerrainTask = task_class(episode_len=256)
task_class = getattr(tasks, "GapTerrainTask")
GapTerrainTask = task_class(episode_len=256)

task_class = getattr(tasks, "CustomizedWallTerrainTask1")
CustomizedWallTerrainTask1 = task_class(episode_len=256)
task_class = getattr(tasks, "CustomizedWallTerrainTask2")
CustomizedWallTerrainTask2 = task_class(episode_len=256)
task_class = getattr(tasks, "CustomizedSteppedTerrainTask2")
CustomizedSteppedTerrainTask2 = task_class(episode_len=300)

task_class = getattr(tasks, "CustomizedBiModalTerrainTask1")
CustomizedBiModalTerrainTask1 = task_class(episode_len=256)
task_class = getattr(tasks, "CustomizedBiModalTerrainTask2")
CustomizedBiModalTerrainTask2 = task_class(episode_len=200)
task_class = getattr(tasks, "HillTerrainTask")
HillTerrainTask = task_class(episode_len=200)

task_class = getattr(tasks, "CustomizedSteppedTerrainTask1")
CustomizedSteppedTerrainTask1 = task_class(episode_len=512)

task_class = getattr(tasks, "CustomizedBiModalTerrainTask3")
CustomizedBiModalTerrainTask3 = task_class(episode_len=300)


k = args.task

if k == 0:
    task_name = 'FlatTerrainTask'
    task = FlatTerrainTask
if k == 1:
    task_name = 'RidgedTerrainTask'
    task = RidgedTerrainTask
if k == 2:
    task_name = 'GapTerrainTask'
    task = GapTerrainTask
if k == 3:
    task_name = 'CustomizedWallTerrainTask1'
    task = CustomizedWallTerrainTask1
if k == 4:
    task_name = 'CustomizedWallTerrainTask2'
    task = CustomizedWallTerrainTask2
if k == 5:
    task_name = 'CustomizedSteppedTerrainTask2'
    task = CustomizedSteppedTerrainTask2
if k == 6:
    task_name = 'CustomizedBiModalTerrainTask1'
    task = CustomizedBiModalTerrainTask1
if k == 7:
    task_name = 'CustomizedBiModalTerrainTask2'
    task = CustomizedBiModalTerrainTask2
if k == 8:
    task_name = 'HillTerrainTask'
    task = HillTerrainTask
if k == 9:
    task_name = 'CustomizedSteppedTerrainTask1'
    task = CustomizedSteppedTerrainTask1
if k == 10:
    task_name = 'CustomizedBiModalTerrainTask3'
    task = CustomizedBiModalTerrainTask3

cnt = 0
vec_list = []

print(k, task_name)
adj_data_name = os.path.join("data", args.data_dir, "adj.npy")
feat_data_name = os.path.join("data", args.data_dir, "feat.npy")

attr_init = np.load(feat_data_name, allow_pickle=True)
conn_init = np.load(adj_data_name, allow_pickle=True)

start_idx = args.start_index
end_idx = args.start_index+args.collecting_size
attr_init = attr_init[start_idx:end_idx]
conn_init = conn_init[start_idx:end_idx]

root_dir = args.save_dir
if not os.path.exists(root_dir):
        os.mkdir(root_dir)

fig_root_path = "designs_figures"
fig_root_path = os.path.join(root_dir,fig_root_path)
if not os.path.exists(fig_root_path):
        os.mkdir(fig_root_path)
good_fig_root_path = "good_designs_figures"
good_fig_root_path = os.path.join(root_dir,good_fig_root_path)
if not os.path.exists(good_fig_root_path):
        os.mkdir(good_fig_root_path)
save_path = task_name
save_path = os.path.join(root_dir,save_path)
if not os.path.exists(save_path):
        os.mkdir(save_path)

cnt = 0
dic = {}
dis_list = []
seq_list = []
adj_matrix_list = []
features_list = []
state_list = []

for i in tqdm(range(len(attr_init))):
    # (adj_matrix_np, features_np) enumerate(zip(conn_init,attr_init))
    adj_matrix_np = conn_init[i]
    features_np = attr_init[i]
    # score = dis_init[i]
    adj_matrix_list.append(adj_matrix_np)
    features_list.append(features_np)

    robot = graph_to_robot(adj_matrix_np, features_np)
    # dof_count = get_dof_count(robot, task, opt_seed, args.jobs, 1)
    score, _, _, input_sequence, _, dof_count, state_np = get_distance(robot, task)
    if dof_count in dic:
        dic[dof_count] += 1
    else:
        dic[dof_count] = 1

    save_fig_path = os.path.join(fig_root_path,"design_iter{}_dof{}_score{}_task{}.png".format(i, dof_count,"{:.0f}".format(score*100), task_name))
    plt.imsave(save_fig_path, get_robot_image(robot, task))
    # print(state_np.shape)
    state_list.append(state_np)
    dis_list.append(score)
    seq_list.append(input_sequence)
    print(dic)
    if score > 1.5:
        save_fig_path = os.path.join(good_fig_root_path,"g_design_iter{}_dof{}_score{}_task{}.png".format(i, dof_count,  "{:.0f}".format(score*100), task_name))
        plt.imsave(save_fig_path, get_robot_image(robot, task))
        # cnt += 1

    if i%10 == 0:
        adj_matrix_np = np.empty(len(adj_matrix_list), dtype=object)
        adj_matrix_np[:] = adj_matrix_list
        features_np = np.empty(len(features_list), dtype=object)
        features_np[:] = features_list
        seq_np = np.empty(len(seq_list), dtype=object)
        seq_np[:] = seq_list
        np.save(os.path.join(save_path, "adj_{}_progress.npy".format(task_name)), adj_matrix_np)
        np.save(os.path.join(save_path, "feat_{}_progress.npy".format(task_name)), features_np)
        np.save(os.path.join(save_path, "dis_{}_profress.npy".format(task_name)), dis_list) 
        np.savez(os.path.join(save_path, "state_{}_progress.npz".format(task_name)), *state_list)  
        np.save(os.path.join(save_path, "seq_{}_progress.npy".format(task_name)), seq_np)     

dis_np = np.stack(dis_list)
print(dis_np.shape)

adj_matrix_np = np.empty(len(adj_matrix_list), dtype=object)
adj_matrix_np[:] = adj_matrix_list
features_np = np.empty(len(features_list), dtype=object)
features_np[:] = features_list
seq_np = np.empty(len(seq_list), dtype=object)
seq_np[:] = seq_list

np.save(os.path.join(save_path, "{}_adj.npy".format(task_name)), adj_matrix_np)
np.save(os.path.join(save_path, "{}_feat.npy".format(task_name)), features_np)
np.save(os.path.join(save_path, "{}_dis.npy".format(task_name)), dis_np) 
np.savez(os.path.join(save_path, "{}_state.npz".format(task_name)), *state_list) 
np.save(os.path.join(save_path, "{}_seq.npy".format(task_name)), seq_np)   


max_l_len = max(dic.keys())
l = [0]*(max_l_len+1)
for k in dic.keys():
    l[k] = dic[k]
print(l)
print(cnt)
print(dic)
plt.bar(list(range(len(l))), l)
plt.show()