import argparse
from email.mime import image
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

# from uniform_reward_net_total_onehot import RewardNet_onehot

from ga_optimization import *

import matplotlib.pyplot as plt

from tqdm import tqdm
# from uniform_reward_net_total_onehot import RewardNet_onehot, MyDataset, train_one_epoch, val_one_epoch

from sklearn.model_selection import train_test_split

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=51, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument('--clip_norm', type=float, default=50.0)
# robot
parser.add_argument("--task", type=str, default="CustomizedBiModalTask1", help="Task (Python class name)")
parser.add_argument("-l", "--episode_len", type=int, default=512,
                        help="Length of episode")
parser.add_argument('--encode', type=str, default="sum")
parser.add_argument("--pred", default=True, action="store_false")

parser.add_argument('--model', type=str, default="sum_ls28_pred20/model.iter-400000")
# parser.add_argument('--model', type=str, default="sum_ls28_pred20_rw_short_v2/model_best")

opt = parser.parse_args()
print(opt)



cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

latent_space = 28

# define range for input
# bounds = [[-2.96, 2.96] for _ in range(28)]
# # define the total iterations
# n_iter = 5
# # bits per variable
# n_bits = 16
# # define the population size
# n_pop = 100
# # crossover rate
# r_cross = 0.9
# # mutation rate
# r_mut = 1.0 / (float(n_bits) * len(bounds))


num_var = 28       # number of decicion variables

bds = np.load("bounds_500k.npy")
varmin = bds[:,0]
varmax = bds[:,1]

# GA Parameters
maxit = 100                                              # number of iterations
npop = 20                                                # initial population size
beta = 1
prop_children = 1                                        # proportion of children to population
num_children = int(np.round(prop_children * npop/2)*2)   # making sure it always an even number
mu = 0.2                                                 # mutation rate 20%, 205 of 5 is 1, mutating 1 gene
sigma = 0.1   

batch_size = 8

model = JTNNVAE(args.hidden_size, args.latent_size, args.depthT, args.encode, args.pred)
# model = JTNNVAE_RW(args.hidden_size, args.latent_size, args.depthT, args.encode, args.pred)
model.load_state_dict(torch.load(args.model))
# model.load_state_dict(torch.load("sum_ls28_pred20_rw_short_v2/model_best"))
model = model.cuda()

# reward_model = RewardNet_onehot(input_length = 28, n_hidden_layers=2, hidden_layer_size = 128, env_size = 9).to(device)
reward_model = JTNNVAE_DIS(128, 32, 80, "sum", True).cuda()
reward_learning_rate = 0.002
reward_weight_decay = 5e-6
reward_optimizer = optim.Adam(reward_model.parameters(), lr=reward_learning_rate, weight_decay=reward_weight_decay)

# ckpt = torch.load('ckpt_exp/uniform_reward_net_500k_1k_4envs_onehot_best.pt')
ckpt = torch.load('vae_train_rw_dis_model_10k/best_model_40.pth')
reward_model.load_state_dict(ckpt['model_state_dict'])

# reward_optimizer.load_state_dict(ckpt['optimizer_state_dict'])

num_envs = 9 

actual_mode = True
if actual_mode:
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
else:
    task_class = getattr(tasks, "FlatTerrainTask")
    FlatTerrainTask = task_class(episode_len=8)
    task_class = getattr(tasks, "RidgedTerrainTask")
    RidgedTerrainTask = task_class(episode_len=8)
    task_class = getattr(tasks, "GapTerrainTask")
    GapTerrainTask = task_class(episode_len=8)

    task_class = getattr(tasks, "CustomizedWallTerrainTask1")
    CustomizedWallTerrainTask1 = task_class(episode_len=8)
    task_class = getattr(tasks, "CustomizedWallTerrainTask2")
    CustomizedWallTerrainTask2 = task_class(episode_len=8)
    task_class = getattr(tasks, "CustomizedSteppedTerrainTask2")
    CustomizedSteppedTerrainTask2 = task_class(episode_len=8)

    task_class = getattr(tasks, "CustomizedBiModalTerrainTask1")
    CustomizedBiModalTerrainTask1 = task_class(episode_len=8)
    task_class = getattr(tasks, "CustomizedBiModalTerrainTask2")
    CustomizedBiModalTerrainTask2 = task_class(episode_len=8)
    task_class = getattr(tasks, "HillTerrainTask")
    HillTerrainTask = task_class(episode_len=8)

    task_class = getattr(tasks, "CustomizedSteppedTerrainTask1")
    CustomizedSteppedTerrainTask1 = task_class(episode_len=8)

    task_class = getattr(tasks, "CustomizedBiModalTerrainTask3")
    CustomizedBiModalTerrainTask3 = task_class(episode_len=8)

def get_env_image(robot, task):
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
        viewer.camera_params.yaw = -np.pi / 2
        viewer.camera_params.pitch = -np.pi / 2
        viewer.camera_params.distance = 6.0
        

    viewer.update(task.time_step)
    
    return viewer.render_array(sim)[:, 500:780, 0].reshape(1, 720, 280)/255

terrain_array_dict = {0 : get_env_image(None, FlatTerrainTask),
                          1 : get_env_image(None, RidgedTerrainTask),
                          2 : get_env_image(None, GapTerrainTask),
                          3 : get_env_image(None, CustomizedWallTerrainTask1),
                          4 : get_env_image(None, CustomizedWallTerrainTask2),
                          5 : get_env_image(None, CustomizedSteppedTerrainTask2),
                          6 : get_env_image(None, CustomizedBiModalTerrainTask1),
                          7 : get_env_image(None, CustomizedBiModalTerrainTask2),
                          8 : get_env_image(None, HillTerrainTask),

                          9 : get_env_image(None, CustomizedSteppedTerrainTask1),
                          10 : get_env_image(None, CustomizedBiModalTerrainTask3),
                        }
    
def get_reward_net_train_data():
    flat_dic = np.load("uniform_flat_dic_500k_1000_400k_v1.npy", allow_pickle=True).item()
    ridged_dic = np.load("uniform_ridged_dic_500k_1000_400k_v1.npy", allow_pickle=True).item()
    gap_dic = np.load("uniform_gap_dic_500k_1000_400k_v1.npy", allow_pickle=True).item()
    cwt1_dic = np.load("uniform_cwt1_256_dic_500k_1000_400k_epi256_v1.npy", allow_pickle=True).item()
    cwt2_dic = np.load("uniform_cwt2_256_dic_500k_1000_400k_epi256_v1.npy", allow_pickle=True).item()
    cst2_dic = np.load("uniform_cst2_dic_500k_1000_400k_epi300_v2.npy", allow_pickle=True).item()
    cbmt1_dic = np.load("uniform_cbmt1_dic_500k_1000_400k_epi256_v2.npy", allow_pickle=True).item()
    cbmt2_dic = np.load("uniform_cbmt2_dic_500k_1000_400k_epi200_v2.npy", allow_pickle=True).item()
    ht_dic = np.load("uniform_ht_dic_500k_1000_400k_epi200_v2.npy", allow_pickle=True).item()

    X = np.concatenate((flat_dic['x'].reshape(-1,28), ridged_dic['x'].reshape(-1,28), gap_dic['x'].reshape(-1,28),
                        cwt1_dic['x'].reshape(-1,28), cwt2_dic['x'].reshape(-1,28), cst2_dic['x'].reshape(-1,28),
                        cbmt1_dic['x'].reshape(-1,28), cbmt2_dic['x'].reshape(-1,28), ht_dic['x'].reshape(-1,28),))
    Y = np.concatenate((ridged_dic['y'], ridged_dic['y'], gap_dic['y'], 
                        cwt1_dic['y'], cwt2_dic['y'], cst2_dic['y'],
                        cbmt1_dic['y'], cbmt2_dic['y'], ht_dic['y'],))

    print(X.shape)
    print(Y.shape)
    # X_env_flat = np.array([0]*flat_dic['x'].shape[0])
    number_of_train_envs = 9
    X_env_flat = np.zeros((flat_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    X_env_flat[:,0] += 1
    X_env_ridged =  np.zeros((ridged_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    X_env_ridged[:,1] += 1
    X_env_gap = np.zeros((gap_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    X_env_gap[:,2] += 1

    X_env_cwt1 = np.zeros((cwt1_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    X_env_cwt1[:,3] += 1
    X_env_cwt2 = np.zeros((cwt2_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    X_env_cwt2[:,4] += 1
    X_env_cst2 = np.zeros((cst2_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    X_env_cst2[:,5] += 1

    X_env_cbmt1 = np.zeros((cbmt1_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    X_env_cbmt1[:,6] += 1
    X_env_cbmt2 = np.zeros((cbmt2_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    X_env_cbmt2[:,7] += 1
    X_env_ht = np.zeros((ht_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    X_env_ht[:,8] += 1
    
    X_env = np.concatenate((X_env_flat, X_env_ridged, X_env_gap,
                            X_env_cwt1, X_env_cwt2, X_env_cst2,
                            X_env_cbmt1, X_env_cbmt2, X_env_ht,))

    print(X.shape, X_env.shape)
    perm = np.random.RandomState(seed=1111).permutation(X.shape[0])
    X = X[perm]
    Y = Y[perm]
    X_env = X_env[perm]
    R_X_train, R_X_val, R_y_train, R_y_val, R_X_env_train, R_X_env_val = train_test_split(
        X, Y, X_env, test_size=0.05, random_state=42)
    print(R_X_train.shape, R_X_env_train.shape)
    return torch.as_tensor(R_X_train).to(device), torch.as_tensor(R_X_val).to(device), \
           torch.as_tensor(R_y_train).to(device), torch.as_tensor(R_y_val).to(device), \
           torch.as_tensor(R_X_env_train).to(device), torch.as_tensor(R_X_env_val).to(device) 

def estimate_reward(X,k_one_hot):
    # X = torch.Tensor(x).to(device)
    # x_env = np.zeros((X.shape[0], num_envs))
    # # x_env = np.zeros((batch_size, 1, 720, 280))
    # x_env[:,k] += 1
    # # x_env[:] = terrain_array_dict[k]
    print(X.shape)
    Y = []
    adj_list = []
    feat_list = []
    for x in X:
        adj_matrix_np, features_np  = decode_graph(model, x.reshape(1,-1))
        adj_list.append(adj_matrix_np)
        feat_list.append(features_np)
    
    train_adj_np = np.empty(len(adj_list), dtype=object)
    train_adj_np[:] = adj_list

    train_feat_np = np.empty(len(feat_list), dtype=object)
    train_feat_np[:] = feat_list

    # print(adj_matrix_np.shape, features_np.shape)
    batch = tensorize(train_feat_np, train_adj_np)

    X_env = torch.Tensor(k_one_hot).to(device)
    # out = reward_model(X, X_env).cpu().data.numpy()
    reward_model.eval()
    out = reward_model(batch)
    # Y.append(out)
    Y = out.cpu().data.numpy()
    # print(Y.shape, np.mean(Y), np.argmax(Y))
    return Y, np.mean(Y), train_adj_np, train_feat_np

def get_distance(robot, task):
        robot_init_pos, has_self_collision = presimulate(robot)

        # finding distance start from here
        if has_self_collision:
            return 0

        def make_sim_fn():
            sim = rd.BulletSimulation(task.time_step)
            task.add_terrain(sim)
            # Rotate 180 degrees around the y axis, so the base points to the right
            sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
            return sim

        main_sim = make_sim_fn()
        robot_idx = main_sim.find_robot_index(robot)

        dof_count = main_sim.get_robot_dof_count(robot_idx)
        
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
                if (j == task.episode_len -1):
                    main_sim.get_robot_world_aabb(robot_idx, lower, upper)
                    # print(lower, upper)
                    # print("diff", lower-init_lower, upper-init_upper)
                    # print("Center diff", (lower+upper)/2 - (init_lower+init_upper)/2)
                    dis = ((lower+upper)/2 - (init_lower+init_upper)/2)[0]
                for k in range(task.interval):
                    main_sim.set_joint_targets(robot_idx,
                                            input_sequence[:, j].reshape(-1, 1))
                    task.add_noise(main_sim, j * task.interval + k)
                    main_sim.step()
                    rewards[j * task.interval + k] = objective_fn(main_sim)
            # value_estimator.get_observation(main_sim, obs[:, -1])
            value_estimator.get_observation(main_sim, np.expand_dims(obs[:, -1], axis=1))

            main_sim.restore_state()
        # camera_params, record_step_indices = view_trajectory(main_sim, robot_idx, input_sequence, task)
        # print("Dis", dis)
        return dis

def get_reward(x, dis_scores, mod_scores, k, epo, i, dir_name, cache, iter):
    adj_matrix_np, features_np  = decode_graph(model, x)
    robot1 = graph_to_robot(adj_matrix_np, features_np)
    if k == 0:
        task = FlatTerrainTask
    if k == 1:
        task = RidgedTerrainTask
    if k == 2:
        task = GapTerrainTask
    if k == 3:
        task = CustomizedWallTerrainTask1
    if k == 4:
        task = CustomizedWallTerrainTask2
    if k == 5:
        task = CustomizedSteppedTerrainTask2
    if k == 6:
        task = CustomizedBiModalTerrainTask1
    if k == 7:
        task = CustomizedBiModalTerrainTask2
    if k == 8:
        task = HillTerrainTask
    result_cache = cache[k]
    result_cache_key = tuple(features_np)
    if result_cache_key in result_cache:
        score = result_cache[result_cache_key]
        print("Hit in env {}".format(k))
    else:

        # seq, score = simulate(robot1, task, opt_seed, 12, 1)
        score = get_distance(robot1, task)
        result_cache[result_cache_key] = score

    fig_root_path = "trainig_figs"
    fig_root_path = os.path.join(dir_name, fig_root_path)
    if not os.path.exists(fig_root_path):
        os.mkdir(fig_root_path)
    save_fig_path = os.path.join(fig_root_path,"Epoch_{}_iter_{}_task_{}_design_{}_{}.png".format(epo, iter, k, i, "{:.0f}".format(score*100)))
    # plt.imshow(get_robot_image(robot1, task), origin='lower')
    # plt.savefig(save_fig_path)
    plt.imsave(save_fig_path, get_robot_image(robot1, task))
    dis_scores.append(score)
    mod_scores.append(features_np.shape[0])
    return score

def multithreading_reward(x, y, idx):
    adj_matrix_np, features_np  = decode_graph(model, x)
    robot1 = graph_to_robot(adj_matrix_np, features_np)
    task_class = getattr(tasks, args.task)
    task = task_class(episode_len=args.episode_len)
    seq, score = simulate(robot1, task, opt_seed, 8, 1)
    y[idx] = score

def get_avg_reward(X, dis_scores, mod_scores, k_list, epo, dir_name, cache, iter):
    # ans = 0
    Y = [0]*len(X)
    for i, (x,k) in enumerate(zip(X, k_list)):
        print("k", k)
        score = get_reward(x.reshape(1,-1), dis_scores, mod_scores, k, epo, i, dir_name, cache, iter)
        Y[i] = score
    out_X = X.clone().detach()
    # print(Y)    
    return sum(Y)/len(Y), out_X.to(device), torch.Tensor(Y).to(device), Y

def get_max_reward(X, idx):
    return get_reward(X[idx].reshape(1,-1).cuda())

def update_reward_net(X, X_env, y, reward_criterion, reward_model=reward_model, \
                      reward_optimizer=reward_optimizer,\
                      reward_data_list=[], running_iter=10):
    train_adj_np, train_feat_np = X
    train_attr_init, train_conn_init, train_loc_init, \
                              val_attr_init, val_conn_init, val_loc_init = reward_data_list
    
    
    best_loss = float("inf")
    # append with old data
    # R_X_train = torch.cat((R_X_train, x))
    # R_y_train = torch.cat((R_y_train, y))
    train_adj_total = np.concatenate((train_conn_init, train_adj_np))
    train_feat_total = np.concatenate((train_attr_init, train_feat_np))
    loc = np.concatenate((train_loc_init, y.cpu().data.numpy()))
    if False:
        R_X_env_train = torch.cat((R_X_env_train, X_env))
        terrain_array_dict = {0 : get_env_image(None, FlatTerrainTask),
                            1 : get_env_image(None, RidgedTerrainTask),
                            2 : get_env_image(None, GapTerrainTask),
                            3 : get_env_image(None, CustomizedWallTerrainTask1),
                            4 : get_env_image(None, CustomizedWallTerrainTask2),
                            5 : get_env_image(None, CustomizedSteppedTerrainTask2),
                            6 : get_env_image(None, CustomizedBiModalTerrainTask1),
                            7 : get_env_image(None, CustomizedBiModalTerrainTask2),
                            8 : get_env_image(None, HillTerrainTask)
                            }
        train_dataset = MyDataset(R_X_train, R_X_env_train, R_y_train, True, terrain_array_dict) # create your datset
        train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True) # create your dataloader



        val_dataset = MyDataset(R_X_val, R_X_env_val, R_y_val, True, terrain_array_dict) # create your datset
        val_dataloader = DataLoader(val_dataset, batch_size = 64, shuffle = False) # create your dataloader
        best_loss = 10
    for i in range(running_iter):
        # train
        cur_loader_idx = 0
        bs = 128
        train_loss = 0
        val_loss = 0
        reward_model.train()
        data_length = len(train_adj_total)
        # loop through the dataset by batch
        while cur_loader_idx < data_length:
            cur_attr = train_feat_total[cur_loader_idx: cur_loader_idx + bs]
            cur_conn = train_adj_total[cur_loader_idx: cur_loader_idx + bs]
            cur_loc = loc[cur_loader_idx: cur_loader_idx + bs]
            cur_loader_idx += bs
            batch = tensorize(cur_attr, cur_conn)
            loc_batch = torch.tensor(cur_loc, dtype=torch.float32).cuda()
            reward_model.zero_grad()
            # print("loc", loc_batch.shape)
            dis = reward_model(batch, loc_batch, beta, 0, 0)
            # print("dis shape", dis.shape, loc_batch.shape)
            loss = reward_criterion(dis, loc_batch)
            train_loss += (loss.item()*len(cur_loc))
            # loss, kl_div, wacc, tacc, pred_loss = model(batch, loc_batch, beta, alpha, gamma)
            loss.backward()
            nn.utils.clip_grad_norm_(reward_model.parameters(), opt.clip_norm)
            reward_optimizer.step()
        cur_loader_idx = 0
        reward_model.eval()
        while cur_loader_idx < len(val_attr_init):
            cur_attr = val_attr_init[cur_loader_idx: cur_loader_idx + batch_size]
            cur_conn = val_conn_init[cur_loader_idx: cur_loader_idx + batch_size]
            # cur_conn = keep_first_one(cur_conn)
            cur_loc = val_loc_init[cur_loader_idx: cur_loader_idx + batch_size]
            cur_loader_idx += batch_size

            batch = tensorize(cur_attr, cur_conn)
            loc_batch = torch.tensor(cur_loc, dtype=torch.float32).cuda()
            # print("loc", loc_batch.shape)
            dis = reward_model(batch, loc_batch, beta, 0, 0)
            # print("dis shape", dis.shape, loc_batch.shape)
            loss = reward_criterion(dis, loc_batch)
            val_loss += (loss.item()*len(cur_loc))

        final_train_loss = train_loss/len(train_attr_init)
        final_val_loss = val_loss/len(val_attr_init)
        print("train loss", final_train_loss, "val loss", final_val_loss)
        # train_loss = train_one_epoch(reward_model, train_dataloader, \
        #                              reward_criterion, reward_optimizer, device)
        # val_loss = val_one_epoch(reward_model, val_dataloader, reward_criterion, device)
        # # for i, batch in enumerate((train_dataloader)):
        # #     reward_model.train()
        # #     reward_optimizer.zero_grad()
        # #     x, x_env, y = batch
        # #     x = x.to(device)
        # #     x_env = x_env.to(device)
        # #     y = y.to(device)

        # #     pred_reward = reward_model(x, X_env).reshape(y.shape)
        # #     reward_loss = reward_criterion(pred_reward.float(), y.float())
        # #     reward_loss.backward(retain_graph=True)
        # #     reward_optimizer.step()
        if final_val_loss < best_loss:
            best_loss = final_val_loss
            reward_model.eval()
            print("Reward Net updated with train loss{}, val loss{}".format(final_train_loss, final_val_loss))
            save_path = "./GAN_ckpt_v34_newVAERW_updateRW_FLAT_only"
            torch.save({'model_state_dict':reward_model.state_dict(),
                        'optimizer_state_dict':reward_optimizer.state_dict()}, os.path.join(save_path, 'reward_net_dis.pt'))

    # return reward_loss.item()/len(y)
    return val_loss


def ga_objective(x,k,reward_model):
	# return x[0]**2.0 + x[1]**2.0 + x[2]**2.0 + x[3]**2.0
    # print(-x[0]**2)
    # return -x[0]**2
    # X = torch.Tensor(x).reshape(1,-1).to(device)
    # x_env = np.zeros((X.shape[0], 2))
    # x_env[:,0] += 1
    # X_env = torch.Tensor(x_env).to(device)
    # return -reward_model(X, X_env).item()
    X = torch.Tensor(x).reshape(1,-1).to(device)
    
    x_env = np.zeros((X.shape[0], num_envs))
    x_env[:,k] += 1
    # x_env = np.zeros((X.shape[0], 1, 720, 280))
    # x_env[:] = terrain_array_dict[k]
    X_env = torch.Tensor(x_env).to(device)

    adj_matrix_np, features_np  = decode_graph(model, X.float().cuda())
    adj_matrix_np = np.expand_dims(adj_matrix_np,0)
    features_np = np.expand_dims(features_np,0)
    # print("double shape",np.concatenate((features_np, features_np), axis=0).shape)
    batch = tensorize(np.concatenate((features_np, features_np), axis=0), np.concatenate((adj_matrix_np, adj_matrix_np), axis=0))
    num_modules = features_np.shape[0]

    # return -reward_model(X, X_env).item(), reward_model(X, X_env).item(), (20 - num_modules)/6 
    return -reward_model(batch)[0].item(), -reward_model(batch)[0].item(), (20 - num_modules)/6 

def data_creation(gen_design,k_list):
    new_data = []
    # print(gen_design.shape, "~~~~~~~~~~~~~")
    for i in range(batch_size):
        k = k_list[i]
        best, score, _, _ = ga_one_iter(ga_objective, num_var, varmin, varmax, 50, 
                npop, num_children, mu, sigma, beta, gen_design[i].cpu().data.numpy(),reward_model,k)

        vec = torch.tensor(best).float().reshape(1, -1).cuda()
        new_data.append(vec)
    new_data = torch.cat(new_data)
    return new_data


class Generator(nn.Module):
    def __init__(self, in_shape=(1, 720, 280),n_channels=4, env_vect_size=20, 
                 env_size = 9, hidden_layer_size = 128):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True,):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(opt.latent_dim+hidden_layer_size//2, 128, normalize=False),
            *block(128, 256),
            # nn.Linear(1024, int(np.prod(img_shape))),
            nn.Linear(256, latent_space),
            nn.Tanh()
        )
        # Terrain conv

        self.activation = torch.nn.ReLU()
        self.embedding_layer = nn.Linear(env_size, hidden_layer_size//2)
        self.bn_conv_in = nn.BatchNorm1d(env_size)

    def forward(self, z, terrain):
        # terrain = terrain.float() 
        # terrain_conv_output = F.relu(self.conv1(terrain))
        # terrain_conv_output = self.bn1(terrain_conv_output)
        # # terrain_conv_output = F.relu(self.conv2(terrain_conv_output))
        # terrain_conv_output = terrain_conv_output.view(-1, self.conv_out_size)
        # terrain_conv_output = self.drops(terrain_conv_output)
        # terrain_conv_output = self.fc(terrain_conv_output)
        
        out = self.bn_conv_in(terrain.float())
        out = self.embedding_layer(out)
        terrain_output = self.activation(out)
        
        x = torch.cat((z, terrain_output), dim=-1)
        
        img = self.model(x)
        # img = img.view(img.size(0), *img_shape)
        # print(img.shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, in_shape=(1, 720, 280),n_channels=4, env_vect_size=20, 
                 latent_space=28, env_size = 9, hidden_layer_size = 128):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(hidden_layer_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        # rb layers
        self.activation_rb = torch.nn.ReLU()
        self.bn_rb = nn.BatchNorm1d(latent_space)
        self.embedding_layer_rb = nn.Linear(latent_space, hidden_layer_size//2)
        
        # Terrain conv
        self.activation = torch.nn.ReLU()
        self.bn_conv_in = nn.BatchNorm1d(env_size)
        self.embedding_layer = nn.Linear(env_size, hidden_layer_size//2)
        

    def forward(self, img, terrain):
        terrain_output = self.activation(self.embedding_layer(self.bn_conv_in(terrain.float())))

        img = self.activation_rb(self.embedding_layer_rb(self.bn_rb(img)))
        img = torch.cat((img, terrain_output), dim=-1)

        # img_flat = img.view(img.size(0), -1)
        # validity = self.model(img_flat)
        validity = self.model(img)
        return validity

def generate_designs(k, GAN_ckpt, epoch, iter, vae_file = "sum_ls28_pred20/model.iter-400000", batch = 8):
    out_dir = os.path.join(GAN_ckpt, "generated_designs_epoch{}".format(epoch))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    generator.load_state_dict(torch.load(os.path.join(GAN_ckpt, "generator_{}_iter_{}.pt".format(epoch, iter))))
    generator.eval()
    # z = Variable(Tensor(np.random.uniform(varmin, varmax, (1, num_var))))
    
    z = Variable(Tensor(np.random.normal(0, 5, (batch, opt.latent_dim))))
    X_env_train = np.zeros((batch, 9))
    
    X_env_train[:,k] += 1
    # X_env_train = np.zeros((batch, 1, 720, 280))
    # X_env_train[:] = terrain_array_dict[k]
    X_env_train = torch.Tensor(X_env_train).to(device)

    out = generator(z, X_env_train)
    print(out.shape)
    vae_model = JTNNVAE(450, 28, 20, "sum", "store_false")
    vae_model.load_state_dict(torch.load(vae_file))

    vae_model = model.cuda()
    result_cache_hit_count = 0
    result_cache = dict()

    GAN_result_dis_list = []
    GAN_result_mod_list = []
    GAN_result_design_list = []
    task_name = ""
    # task_list = ["FlatTerrainTask", "RidgedTerrainTask", "GapTerrainTask", "CustomizedWallTerrainTask2"]

    # task_name = task_list[k]
    # task_class = getattr(tasks, task_name)
    # task = task_class(episode_len=args.episode_len)
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

    for i in tqdm(range(out.shape[0])):
        vec = out[i].reshape(1,-1)
        adj_matrix_np, features_np  = decode_graph(vae_model, vec)

        result_cache_key = tuple(features_np)
        if result_cache_key in result_cache:
            result = result_cache[result_cache_key]
            result_cache_hit_count += 1
        else:
            robot = graph_to_robot(adj_matrix_np, features_np)
            input_sequence, result = simulate(robot, task, opt_seed, 12, 1)
            result_cache[result_cache_key] = result
            fig_root_path = "testing_figs"
            fig_root_path = os.path.join(GAN_ckpt, fig_root_path)
            if not os.path.exists(fig_root_path):
                os.mkdir(fig_root_path)
            save_fig_path = os.path.join(fig_root_path,"Epoch_{}_task_{}_design_{}_{}.png".format(epoch, task_name, i, "{:.0f}".format(result*100)))
            plt.imsave(save_fig_path, get_robot_image(robot, task))

        GAN_result_design_list.append(vec.cpu().data.numpy())
        GAN_result_dis_list.append(result)
        GAN_result_mod_list.append(features_np.shape[0])

    GAN_result_design_np = np.stack(GAN_result_design_list)
    GAN_result_dis_np = np.stack(GAN_result_dis_list)
    GAN_result_mod_np = np.stack(GAN_result_mod_list)
    print(GAN_result_design_np.shape)
    print(GAN_result_dis_np.shape, GAN_result_mod_np.shape)
    print(np.unique(GAN_result_dis_np))
    np.save(os.path.join(out_dir, "GAN_result_dis_500_v4_4env_{}.npy".format(task_name)), GAN_result_dis_np) 
    np.save(os.path.join(out_dir, "GAN_result_mod_500_v4_4env_{}.npy".format(task_name)), GAN_result_mod_np) 
    np.save(os.path.join(out_dir, "GAN_result_design_500_v4_4env_{}.npy".format(task_name)), GAN_result_design_np)

    print(f'iteration:{i}',f"hitting cache, total = {result_cache_hit_count}")

if __name__ == "__main__":
    # Get Reward Net training data
    # R_X_train, R_X_val, R_y_train, R_y_val, R_X_env_train, R_X_env_val  = get_reward_net_train_data()
    # reward_train_data_list = [R_X_train, R_X_val, R_y_train, R_y_val, R_X_env_train, R_X_env_val]
    
    adj_data_name = os.path.join("data", "new_train_data_dis_4k", "train_adj_even.npy")
    feat_data_name = os.path.join("data", "new_train_data_dis_4k", "train_feat_even.npy")
    loc_data_name = os.path.join("data", "new_train_data_dis_4k", "train_dis_even.npy")
    print(f"loading data from {adj_data_name}")
    train_attr_init = np.load(feat_data_name, allow_pickle=True)
    train_conn_init = np.load(adj_data_name, allow_pickle=True)
    train_loc_init = np.load(loc_data_name, allow_pickle=True)
    
    adj_data_name = os.path.join("data", "new_train_data_dis_4k", "val_adj_even.npy")
    feat_data_name = os.path.join("data", "new_train_data_dis_4k", "val_feat_even.npy")
    loc_data_name = os.path.join("data", "new_train_data_dis_4k", "val_dis_even.npy")
    print(f"loading data from {adj_data_name}")
    val_attr_init = np.load(feat_data_name, allow_pickle=True)
    val_conn_init = np.load(adj_data_name, allow_pickle=True)
    val_loc_init = np.load(loc_data_name, allow_pickle=True)
    

    reward_train_data_list = [train_attr_init, train_conn_init, train_loc_init, 
                              val_attr_init, val_conn_init, val_loc_init]

    # print("R x env train shape",R_X_env_train.shape)
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    generator = generator.to(device)
    # generator.load_state_dict(torch.load(os.path.join("GAN_ckpt_v33_epoch11_3envs_uniform_total_onehot_lb_bestR_warmupgen", 
    #                                                   "generator_{}_iter_{}.pt".format(20, 0))))
    discriminator = discriminator.to(device)
    adversarial = adversarial_loss.to(device)
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    estimated_rewards = []
    actual_rewards = []
    D_losses = []
    G_losses = []
    reward_loss = []
    
    number_of_envs = 9

    global_cache = dict()
    for k_idx in range(number_of_envs+1):
        global_cache[k_idx] = dict()

    axis = []
    for i in range(opt.n_epochs):
        axis.append(50*i)
    
    train = True
    generate = False
    visualize = False
    # # ----------
    # #  Training
    # # ----------
    if train:
        iteration = 50
        ckpt_path = "GAN_ckpt_v37_newVAERW_RW_FLAT_small_long_only"
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        reward_criterion = torch.nn.MSELoss()
        dis_scores = []
        mod_scores = []
        real_dis_scores = [] 
        real_mod_scores = []
        fake_avg_reward = []
        real_avg_reward = []
        fake_est_reward = []
        real_est_reward = []
        envs = []
        env_dim = (1,720,280)
        for epoch in tqdm(range(opt.n_epochs)):
            for i in range(iteration):
                print("I:", i)
                # Adversarial ground truths
                valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

                X_env_train = np.zeros((batch_size, number_of_envs))
                # k = random.randint(0, 3)
                # k = epoch%3
                # k = 3

                # Generate random list where env is econded as terrain map
                # k = random.randint(0,8)
                # X_env_train[:,k] += 1
                # X_env_train = np.zeros((batch_size, 1, 720, 280))
                # X_env_train[:,] = terrain_array_dict[k]
                # X_env_train = torch.Tensor(X_env_train).to(device)
                
                # Generate random list of k, the env is one hot encoded 
                # k_list = np.arange(batch_size)
                # np.random.shuffle(k_list)
                # k_list[k_list >= 9] %= 9
                
                # k_one_hot = np.zeros((k_list.size, k_list.max()+1))
                # k_one_hot[np.arange(k_list.size), k_list] = 1

                # Generate env using only flat with one hot encoded
                k_list = [0]*batch_size
                k_one_hot = np.zeros((batch_size, 9))
                k_one_hot[:,0] = 1

                X_env_train = torch.Tensor(k_one_hot).to(device)
                
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 5, (batch_size, opt.latent_dim)))).to(device)

                # Generate a batch of images
                gen_imgs = generator(z, X_env_train)
                

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gen_imgs, X_env_train), valid)
                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                if i % 1 == 0:
                    out = gen_imgs
                    pred_reward, mean_reward, train_adj_np, train_feat_np = estimate_reward(out, k_one_hot)
                    print("Genrated", pred_reward)

                    # print("mean_reward", mean_reward)
                    fake_est_reward.append(mean_reward)
                    # actual_reward = get_max_reward(out, max_idx)
                    
                    # # actual gen-image rewards
                    # actual_reward, reward_train_X, reward_train_Y, gen_sim_rewards = \
                    #     get_avg_reward(out, dis_scores, mod_scores, k_list, epoch,ckpt_path, global_cache, i)
                    
                    # fake_avg_reward.append(actual_reward)

                    estimated_rewards.append(np.mean(pred_reward))
                    # actual_rewards.append(actual_reward)
                    # loss_r = update_reward_net((train_adj_np, train_feat_np), X_env_train, \
                    #                            reward_train_Y, reward_criterion, \
                    #                            reward_model=reward_model,\
                    #                            reward_data_list=reward_train_data_list, running_iter=10)
                    
                    # evolved image rewards
                    # evo_imgs = data_creation(gen_imgs, k_list)
                    # print("Evo shape",evo_imgs.shape)
                    # evo_actual_reward, evo_reward_train_X, \
                    # evo_reward_train_Y, evo_gen_sim_rewards = \
                    #     get_avg_reward(evo_imgs, [], [], k_list, \
                    #                    epoch,ckpt_path, global_cache, i)
    #                 loss_r = update_reward_net(evo_reward_train_X, X_env_train, \
    #                                            evo_reward_train_Y, reward_criterion, \
    #                                            reward_model=reward_model,\
    #                                            reward_data_list=reward_train_data_list, running_iter=10)
                    
                    gen_labels = torch.Tensor(batch_size, 1).fill_(0.0)
                    real_labels = torch.Tensor(batch_size, 1).fill_(1.0)
    #                 for inner_i, (gen_x, gen_y, evo_x, evo_y) in enumerate(zip(gen_imgs, gen_sim_rewards,\
    #                                                                      evo_imgs, evo_gen_sim_rewards)):
    #                     if evo_y > gen_y:
    #                         real_labels[inner_i] = 1.0
    #                     else:
    #                         gen_labels[inner_i] = 1.0
                    
                    valid = Variable(real_labels, requires_grad=False).to(device)
                    fake = Variable(gen_labels, requires_grad=False).to(device)
                    
                    # reward_loss.append(loss_r)
                    print("Epoch: {}, Iter: {}, estimated reward".format(epoch, i), np.mean(pred_reward))
                    # print("Epoch: {}, Iter: {}, acutal reward".format(epoch, i), actual_reward)
                    print(dis_scores)
                    print(mod_scores)
                # Configure input, Evolve the designs
                
                # reward_model.eval()
                real_imgs = data_creation(gen_imgs, k_list)
                pred_reward, mean_reward, train_adj_np, train_feat_np = estimate_reward(real_imgs, k_one_hot)
                print("Evolved",pred_reward)

                # actual_reward, reward_train_X, reward_train_Y, gen_sim_rewards = \
                #         get_avg_reward(real_imgs, real_dis_scores, real_mod_scores, k_list, epoch,ckpt_path, global_cache, i)
                # loss_r = update_reward_net((train_adj_np, train_feat_np), X_env_train, \
                #                                reward_train_Y, reward_criterion, \
                #                                reward_model=reward_model,\
                #                                reward_data_list=reward_train_data_list, running_iter=10)    
                real_est_reward.append(mean_reward)
                # real_avg_reward.append(actual_reward)

                print("The list", real_avg_reward, real_est_reward, fake_avg_reward, fake_est_reward)
                # np.save(ckpt_path+"/real_avg_reward.npy", np.array(real_avg_reward))
                np.save(ckpt_path+"/real_est_reward.npy", np.array(real_est_reward))
                # np.save(ckpt_path+"/fake_avg_reward.npy", np.array(fake_avg_reward))
                np.save(ckpt_path+"/fake_est_reward.npy", np.array(fake_est_reward))
                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_imgs, X_env_train), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), X_env_train), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, iteration, d_loss.item(), g_loss.item())
                )
                D_losses.append(d_loss.item())
                G_losses.append(g_loss.item())
            torch.save(generator.state_dict(), os.path.join(ckpt_path, "generator_{}_iter_{}.pt".format(epoch, iteration)))
            torch.save(discriminator.state_dict(), os.path.join(ckpt_path, "discriminator_{}_iter_{}.pt".format(epoch, iteration)))

        dis_scores = np.stack(dis_scores, axis=0)
        mod_scores = np.stack(mod_scores, axis=0)
        np.save(os.path.join(ckpt_path, f"dis_scores.npy"), np.array(dis_scores))
        np.save(os.path.join(ckpt_path, f"mod_scores.npy"), np.array(mod_scores))

        np.save(os.path.join(ckpt_path, f"estimated.npy"), np.array(estimated_rewards))
        np.save(os.path.join(ckpt_path, f"simulated.npy"), np.array(actual_rewards))
        np.save(os.path.join(ckpt_path, f"g_losses.npy"), np.array(G_losses))
        np.save(os.path.join(ckpt_path, f"d_losses.npy"), np.array(D_losses))
        plt.plot(axis, estimated_rewards, color='r', label='Reward Net')
        plt.plot(axis, actual_rewards, color='b', label="Simulation")
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        print(envs)
        # envs_np = np.stack(envs)
        # np.save(os.path.join(ckpt_path,"envs.npy"), envs_np)

        plt.show()
    # # ----------
    # #  Generating
    # # ----------
    # if generate:
        
    #     # for i in range(9):
    #     #     generate_designs(i, 'GAN_ckpt_v29_epoch11_3envs_uniform_total_onehot_lb', 10, vae_file = "sum_ls28_pred20/model.iter-400000",batch = 10)

    #     # for ep in range(11):
    #     #     for i in range(9):
    #     #         generate_designs(i, 'GAN_ckpt_v32_epoch11_3envs_uniform_total_onehot_lb_bestR_relabel_warmstart', ep, vae_file = "sum_ls28_pred20/model.iter-400000",batch = 10)
    #     for i in range(9):
    #         generate_designs(i, 'GAN_ckpt_v33_epoch11_3envs_uniform_total_onehot_lb_bestR_relabel_warmstart_sim', 4, 50, vae_file = "sum_ls28_pred20/model.iter-400000",batch = 10)
