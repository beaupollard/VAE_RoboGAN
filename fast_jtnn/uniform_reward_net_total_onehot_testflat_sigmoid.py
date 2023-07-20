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
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils import data
import torch.nn.functional as F

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
        viewer.camera_params.yaw = -np.pi / 2
        viewer.camera_params.pitch = -np.pi / 2
        viewer.camera_params.distance = 6.0
        

    viewer.update(task.time_step)
    
    return viewer.render_array(sim)[:, 500:780, 0].reshape(1, 720, 280)/255

class RewardNet_onehot_sigmoid(nn.Module):
    def __init__(self, input_length: int, n_hidden_layers=3, hidden_layer_size = 128, env_size = 9, 
                 in_shape=(1, 720, 280),n_channels=4, env_vect_size=20, latent_layers=[128,256,512,128]):
        super(RewardNet_onehot_sigmoid, self).__init__()

        # preprocessing - can be changed
        self.input_layer = nn.Linear(int(input_length), hidden_layer_size//2)
        self.embedding_layer = nn.Linear(env_size, hidden_layer_size//2)

        #hidden layer
        self.hidden_list = torch.nn.ModuleList()
        self.n_hidden_layers = n_hidden_layers
        for i in range(n_hidden_layers):
            in_size = latent_layers[i]
            out_size = latent_layers[i+1]
            self.hidden_list.append(nn.Linear(in_size, out_size))
        self.output_layer = nn.Linear(latent_layers[-1], 1)
        self.activation = torch.nn.ReLU()

        self.bn_list = torch.nn.ModuleList()
        for i in range(1, n_hidden_layers + 1):
            in_size = latent_layers[i]
            self.bn_list.append(nn.BatchNorm1d(in_size))

        self.drops = nn.Dropout(0.3)
        self.bn_conv_in = nn.BatchNorm1d(env_size)
        self.sigmoid = nn.Sigmoid()
        # Terrain conv
        # self.in_shape = (1, 720, 280)
        # self.conv_out_size = self.in_shape[2]*self.in_shape[1]*n_channels

        # self.conv1 = torch.nn.Conv2d(in_shape[0], n_channels,
        #     kernel_size=5, stride=1, padding=5//2)
        # self.bn1 = nn.BatchNorm2d(n_channels)
        # self.conv2 = torch.nn.Conv2d(n_channels, n_channels,
        #     kernel_size=5, stride=1, padding=5//2)
        # self.bn2 = nn.BatchNorm2d(n_channels)

        # #add dropout?
        # self.drops = nn.Dropout(0.3)
        # self.fc = nn.Linear(self.conv_out_size, env_vect_size)

    def forward(self, robot, terrain):
        # terrain = terrain.float() 
        # terrain_conv_output = F.relu(self.conv1(terrain))
        # terrain_conv_output = self.bn1(terrain_conv_output)
        # terrain_conv_output = F.relu(self.conv2(terrain_conv_output))
        # terrain_conv_output = terrain_conv_output.view(-1, self.conv_out_size)
        # terrain_conv_output = self.drops(terrain_conv_output)
        # terrain_conv_output = self.fc(terrain_conv_output)

        x1 = self.activation(self.input_layer(robot))
        
        # x2 = self.activation(self.embedding_layer(self.bn_conv_in(terrain_conv_output)))
        x2 = self.activation(self.embedding_layer(self.bn_conv_in(terrain.float())))

        x = torch.cat((x1, x2), dim=-1)

        for i in range(self.n_hidden_layers):
            x = self.activation(self.hidden_list[i](x))

            # added dropout and bn: see if it works
            x = self.drops(x)
            x = self.bn_list[i](x)

        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

class MyDataset(data.Dataset):
    def __init__(self, X, X_env, Y, train, env_dict):
        self.X = torch.Tensor(X)
        self.X_env = X_env
        self.train = train
        self.Y = Y#.astype()
        self.env_dict = env_dict
        

    def __len__(self):
        # return self.X.shape[0]
        return len(self.X)

    def __getitem__(self, index):
        # For testing set, return only x
        if self.Y is None:
            return torch.as_tensor(self.X[index]), torch.as_tensor(self.X_env[index])
            # For training and validation set, return x and y
        else:
            # if self.train:
            #     return self.mask(torch.as_tensor(self.X[index])), torch.as_tensor(self.Y[index])
            return torch.as_tensor(self.X[index]), torch.as_tensor(self.X_env[index]), torch.as_tensor(self.Y[index])

def train_one_epoch(model, train_dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for i, batch in enumerate((train_dataloader)):
        
        optimizer.zero_grad()
        x, x_env, y = batch
        # y = normalize_labels(y)
        x = x.to(device)
        x_env = x_env.to(device)
        y = y.to(device)
        
        pred_reward = model(x, x_env).reshape(y.shape)
        loss = criterion(pred_reward.float(), y.float())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    return train_loss/len(train_dataloader)

def train_one_epoch_mini(model, train_dataloader, criterion, optimizer):
    model.train()
    # train_loss = 0
    # for i, batch in enumerate((train_dataloader)):
    #     optimizer.zero_grad()
    #     x, x_env, y = batch
    #     x = x.to(device)
    #     x_env = x_env.to(device)
    #     y = y.to(device)
        
    #     pred_reward = model(x, x_env).reshape(y.shape)
    #     loss = criterion(pred_reward.float(), y.float())
    #     train_loss += loss.item()
    #     loss.backward()
    #     optimizer.step()
    
    # return train_loss/len(train_dataloader)

def val_one_epoch(model, val_dataloader, criterion, device):
    model.eval()
    val_loss = 0
    for i, batch in enumerate((val_dataloader)):
        x, x_env, y = batch
        x = x.to(device)
        x_env = x_env.to(device)
        y = y.to(device)
        
        pred_reward = model(x, x_env).reshape(y.shape)
        # y = normalize_labels(y)
        # pred_reward = unnormalize_labels(pred_reward)
        loss = criterion(pred_reward.float(), y.float())
        val_loss += loss.item()
    
    return val_loss/len(val_dataloader)

# Function for label normalization
def normalize_labels(labels):
    global min_label_val, max_label_val
    min_label_val = min(min_label_val, labels.min().item())
    max_label_val = max(max_label_val, labels.max().item())
    return (labels - min_label_val) / (max_label_val - min_label_val)

# Function for label unnormalization
def unnormalize_labels(normalized_labels):
    return normalized_labels * (max_label_val - min_label_val) + min_label_val


if __name__ == "__main__":
    # Variables for recording min and max values
    min_label_val = float('inf')
    max_label_val = float('-inf')
    
    number_of_envs = 9
    
    # flat_dic = np.load("uniform_flat_dic_500k_1000_400k_v1.npy", allow_pickle=True).item()
    # ridged_dic = np.load("uniform_ridged_dic_500k_1000_400k_v1.npy", allow_pickle=True).item()
    # gap_dic = np.load("uniform_gap_dic_500k_1000_400k_v1.npy", allow_pickle=True).item()
    # cwt1_dic = np.load("uniform_cwt1_256_dic_500k_1000_400k_epi256_v1.npy", allow_pickle=True).item()
    # cwt2_dic = np.load("uniform_cwt2_256_dic_500k_1000_400k_epi256_v1.npy", allow_pickle=True).item()
    # cst2_dic = np.load("uniform_cst2_dic_500k_1000_400k_epi300_v2.npy", allow_pickle=True).item()
    # cbmt1_dic = np.load("uniform_cbmt1_dic_500k_1000_400k_epi256_v2.npy", allow_pickle=True).item()
    # cbmt2_dic = np.load("uniform_cbmt2_dic_500k_1000_400k_epi200_v2.npy", allow_pickle=True).item()
    # ht_dic = np.load("uniform_ht_dic_500k_1000_400k_epi200_v2.npy", allow_pickle=True).item()

    # X = np.concatenate((flat_dic['x'].reshape(-1,28), ridged_dic['x'].reshape(-1,28), gap_dic['x'].reshape(-1,28),
    #                     cwt1_dic['x'].reshape(-1,28), cwt2_dic['x'].reshape(-1,28), cst2_dic['x'].reshape(-1,28),
    #                     cbmt1_dic['x'].reshape(-1,28), cbmt2_dic['x'].reshape(-1,28), ht_dic['x'].reshape(-1,28),))
    # Y = np.concatenate((ridged_dic['y'], ridged_dic['y'], gap_dic['y'], 
    #                     cwt1_dic['y'], cwt2_dic['y'], cst2_dic['y'],
    #                     cbmt1_dic['y'], cbmt2_dic['y'], ht_dic['y'],))
    
    # X = np.load('dis_data/CustomizedFlatTerrainTask_x_aug_balenced_v3.npy')
    # Y = np.load('dis_data/CustomizedFlatTerrainTask_y_aug_balenced_v3.npy')
    # v4 is 10k data.


    # flat_dic = np.load("dis_data/uniform_FlatTerrainTask_dic_500k_1000_400k_epi256_v2.npy", allow_pickle=True).item()
    # X = flat_dic['x'].reshape(-1,28)
    # Y = flat_dic['y']

    # print(X.shape)
    # print(Y.shape)
    # # X_env_flat = np.array([0]*flat_dic['x'].shape[0])
    # # number_of_train_envs = 9
    # # X_env_flat = np.zeros((flat_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    # # X_env_flat[:,0] += 1
    # # X_env_ridged =  np.zeros((ridged_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    # # X_env_ridged[:,1] += 1
    # # X_env_gap = np.zeros((gap_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    # # X_env_gap[:,2] += 1

    # # X_env_cwt1 = np.zeros((cwt1_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    # # X_env_cwt1[:,3] += 1
    # # X_env_cwt2 = np.zeros((cwt2_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    # # X_env_cwt2[:,4] += 1
    # # X_env_cst2 = np.zeros((cst2_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    # # X_env_cst2[:,5] += 1

    # # X_env_cbmt1 = np.zeros((cbmt1_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    # # X_env_cbmt1[:,6] += 1
    # # X_env_cbmt2 = np.zeros((cbmt2_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    # # X_env_cbmt2[:,7] += 1
    # # X_env_ht = np.zeros((ht_dic['x'].reshape(-1,28).shape[0], number_of_train_envs))
    # # X_env_ht[:,8] += 1
    
    # # X_env = np.concatenate((X_env_flat, X_env_ridged, X_env_gap,
    # #                         X_env_cwt1, X_env_cwt2, X_env_cst2,
    # #                         X_env_cbmt1, X_env_cbmt2, X_env_ht,))
    # X_env = np.zeros((X.shape[0], 9))
    # X_env[:,0] += 1
    # print(X.shape, X_env.shape)
    # print(np.sum((Y < 5)))
    # print(np.sum((5 <= Y) & (Y <= 10)))
    # print(np.sum((10 < Y)))
    # # X_train = []
    # # y_train = []
    # # X_val = []
    # # y_val = []

    # # bucket_1 = []
    # # bucket_2 = []
    # # bucket_3 = []

    # # for x, y in zip(X, Y):
    # #     if y < 5:
    # #         bucket_1.append((x, y))
    # #     elif y < 10:
    # #         bucket_2.append((x, y))
    # #     else:
    # #         bucket_3.append((x, y))

    # # bucket_1_train_len = int(0.85 * len(bucket_1))
    # # bucket_2_train_len = int(0.85 * len(bucket_2))
    # # bucket_3_train_len = int(0.85 * len(bucket_3))

    # # # Bucket 1
    # # X_train.extend([x for x, _ in bucket_1[:bucket_1_train_len]])
    # # y_train.extend([y for _, y in bucket_1[:bucket_1_train_len]])
    # # X_val.extend([x for x, _ in bucket_1[bucket_1_train_len:]])
    # # y_val.extend([y for _, y in bucket_1[bucket_1_train_len:]])

    # # # Bucket 2
    # # X_train.extend([x for x, _ in bucket_2[:bucket_2_train_len]])
    # # y_train.extend([y for _, y in bucket_2[:bucket_2_train_len]])
    # # X_val.extend([x for x, _ in bucket_2[bucket_2_train_len:]])
    # # y_val.extend([y for _, y in bucket_2[bucket_2_train_len:]])

    # # # Bucket 3
    # # X_train.extend([x for x, _ in bucket_3[:bucket_3_train_len]])
    # # y_train.extend([y for _, y in bucket_3[:bucket_3_train_len]])
    # # X_val.extend([x for x, _ in bucket_3[bucket_3_train_len:]])
    # # y_val.extend([y for _, y in bucket_3[bucket_3_train_len:]])
    # # X_train = np.array(X_train)
    # # y_train = np.array(y_train)
    # # X_env_train = X_env[:len(X_train)]
    # # X_env_val = X_env[len(X_train):]
    # perm = np.random.RandomState(seed=42).permutation(X.shape[0])
    # X = X[perm]
    # Y = Y[perm]
    # X_env = X_env[perm]
    # X_train, X_val, y_train, y_val, X_env_train, X_env_val = train_test_split(
    #     X, Y, X_env, test_size=0.1, random_state=42)
    
    # below is new vae latent data
    X_train = np.load('dis_data/new_1k_X_train_flat.npy')
    y_train = np.load('dis_data/new_1k_y_train_flat.npy')
    X_val = np.load("dis_data/new_1k_X_val_flat.npy")
    y_val = np.load("dis_data/new_1k_Y_val_flat.npy")
    X_env_train = np.zeros((X_train.shape[0], 9))
    X_env_train[:,0] += 1
    X_env_val = np.zeros((X_val.shape[0], 9))
    X_env_val[:,0] += 1
    # print(X_train.shape, X_env_train.shape, np.sum(y_train > 10)/np.sum(Y > 10),
    #       np.sum((5<= y_train)&(y_train <= 10))/ np.sum((5 <= Y) & (Y <= 10)), )

    # print(sum(Y>10))
    # task_class = getattr(tasks, "FlatTerrainTask")
    # FlatTerrainTask = task_class(episode_len=256)
    # task_class = getattr(tasks, "RidgedTerrainTask")
    # RidgedTerrainTask = task_class(episode_len=256)
    # task_class = getattr(tasks, "GapTerrainTask")
    # GapTerrainTask = task_class(episode_len=256)

    # task_class = getattr(tasks, "CustomizedWallTerrainTask1")
    # CustomizedWallTerrainTask1 = task_class(episode_len=256)
    # task_class = getattr(tasks, "CustomizedWallTerrainTask2")
    # CustomizedWallTerrainTask2 = task_class(episode_len=256)
    # task_class = getattr(tasks, "CustomizedSteppedTerrainTask2")
    # CustomizedSteppedTerrainTask2 = task_class(episode_len=300)

    # task_class = getattr(tasks, "CustomizedBiModalTerrainTask1")
    # CustomizedBiModalTerrainTask1 = task_class(episode_len=256)
    # task_class = getattr(tasks, "CustomizedBiModalTerrainTask2")
    # CustomizedBiModalTerrainTask2 = task_class(episode_len=200)
    # task_class = getattr(tasks, "HillTerrainTask")
    # HillTerrainTask = task_class(episode_len=200)

    # terrain_array_dict = {0 : get_robot_image(None, FlatTerrainTask),
    #                       1 : get_robot_image(None, RidgedTerrainTask),
    #                       2 : get_robot_image(None, GapTerrainTask),
    #                       3 : get_robot_image(None, CustomizedWallTerrainTask1),
    #                       4 : get_robot_image(None, CustomizedWallTerrainTask2),
    #                       5 : get_robot_image(None, CustomizedSteppedTerrainTask2),
    #                       6 : get_robot_image(None, CustomizedBiModalTerrainTask1),
    #                       7 : get_robot_image(None, CustomizedBiModalTerrainTask2),
    #                       8 : get_robot_image(None, HillTerrainTask)
    #                     }
    
    train_dataset = MyDataset(X_train, X_env_train, y_train, True, None) # create your datset
    train_dataloader = DataLoader(train_dataset, batch_size = 512, shuffle = True) # create your dataloader



    val_dataset = MyDataset(X_val, X_env_val, y_val, True, None) # create your datset
    val_dataloader = DataLoader(val_dataset, batch_size = 512, shuffle = False) # create your dataloader
    
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    epoch = 100
    model = RewardNet_onehot(input_length = X_train.shape[1], n_hidden_layers=1, hidden_layer_size = 128,
                             latent_layers=[128,128])
    model = model.to(device)
    print(model)
    learning_rate = 0.002
    weight_decay = 5e-6
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    train_losses = []
    val_losses = []
    best_loss = 10

    for inputs, x_env, labels in train_dataloader:
        min_label_val = min(min_label_val, labels.min().item())
        max_label_val = max(max_label_val, labels.max().item())

    for i in range(epoch):
        # train
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        train_losses.append(train_loss)
        # val
        val_loss = val_one_epoch(model, val_dataloader, criterion, device)
        val_losses.append(val_loss)
        print(train_loss, val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            print("Current Best Loss is:", best_loss, "Train Loss: {}".format(train_loss))
            save_path = "./ckpt_exp"
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict()}, os.path.join(save_path, 'uniform_reward_net_500k_1k_4envs_onehot_flatonly_test.pt'))
    ckpt = torch.load(os.path.join(save_path, 'uniform_reward_net_500k_1k_4envs_onehot_flatonly_test.pt'))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    val_loss = 0
    print("validation result")
    pred_list = []
    gt_list = []
    for i, batch in enumerate((val_dataloader)):
        x, x_env, y = batch
        x = x.to(device)
        x_env = x_env.to(device)
        y = y.to(device)
        
        pred_reward = model(x, x_env).reshape(y.shape)
        for j, (p,g) in enumerate(zip(pred_reward, y)):
            print(p.item(), g.item())
            pred_list.append(p.item())
            gt_list.append(g.item())
            # plt.plot([j,j], [p.item(), g.item()])
        # y = normalize_labels(y)
        # pred_reward = unnormalize_labels(pred_reward)
        # loss = criterion(pred_reward.float(), y.float())
        # val_loss += loss.item()
    axis = np.arange(len(pred_list))

    sorted_indices = np.argsort(pred_list)
    sorted_another_list = [gt_list[i] for i in sorted_indices]
    for i, (p,g) in enumerate(zip(sorted(pred_list), sorted_another_list)):
        plt.plot([i,i], [p, g])

    plt.scatter(axis, sorted(pred_list), c='b', label="pred value")
    plt.scatter(axis, sorted_another_list, c='g', label="ground truth value")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.ylabel('distance')
    plt.xlabel('data ID')
    plt.show()
    plt.close()
    plt.plot(range(len(train_losses)), train_losses, label="train loss", color='r')
    plt.plot(range(len(val_losses)), val_losses, label="val loss", color='b')   
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    
    # plt.ylim([0, 20])
    plt.show()  
    # save_path = "./ckpt_exp"
    # torch.save({'model_state_dict':model.state_dict(),
    #             'optimizer_state_dict':optimizer.state_dict()}, os.path.join(save_path, 'uniform_reward_net_500k_1k_4envs_onehot_test.pt'))
