'''
For training the generative model that encodes designs
example:
python train_gnn_reward_net.py --save_dir vae_rw_dis10k_cwt1 --data_dir new_train_data_dis_2k_cwt1 --gamma 20
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os

import math, sys
import numpy as np
import argparse
from fast_jtnn import *
import matplotlib.pyplot as plt
import copy

from robot_utils import *
import pyrobotdesign as rd
import robot_utils.tasks as tasks
from tqdm import tqdm
def keep_first_one(ms):
    total = []
    for matrix in ms:
        for row in matrix:
            found_one = False
            for i in range(len(row)):
                if row[i] == 1:
                    if found_one:
                        row[i] = 0
                    else:
                        found_one = True
        total.append(matrix)
    return total

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

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', required=True)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--load_epoch', type=int, default=0)

# parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=128)
# parser.add_argument('--latent_size', type=int, default=28)
parser.add_argument('--latent_size', type=int, default=32) # 32 was tahe best

# parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthT', type=int, default=80)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--step_beta', type=float, default=0.002)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=40000)

parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=40000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=5000)

parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--gamma', type=float, required=True)
parser.add_argument('--encode', type=str, default="sum")
parser.add_argument('--pred', default=True, action="store_false")


args = parser.parse_args()
print(args)


model = JTNNVAE_DIS(args.hidden_size, args.latent_size, args.depthT, args.encode, args.pred).cuda()
print(model)


for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay= 1e-4)
optimizer = optim.SGD(model.parameters(), lr = 1e-2, weight_decay=1e-4, momentum=0.9)
# scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
# scheduler.step()
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=10, verbose=True)

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = args.load_epoch
beta = args.beta
alpha = args.alpha
gamma = args.gamma
meters = np.zeros(4)

train_adj_data_name = os.path.join("data", args.data_dir, "train_adj_even.npy")
train_feat_data_name = os.path.join("data", args.data_dir, "train_feat_even.npy")
train_loc_data_name = os.path.join("data", args.data_dir, "train_dis_even.npy")

print(f"loading data from {train_adj_data_name}")

attr_init = np.load(train_feat_data_name, allow_pickle=True)
conn_init = np.load(train_adj_data_name, allow_pickle=True)
loc_init = np.load(train_loc_data_name, allow_pickle=True)


val_adj_data_name = os.path.join("data", args.data_dir, "val_adj_even.npy")
val_feat_data_name = os.path.join("data", args.data_dir, "val_feat_even.npy")
val_loc_data_name = os.path.join("data", args.data_dir, "val_dis_even.npy")

print(f"loading data from {train_feat_data_name}")

val_attr = np.load(val_feat_data_name, allow_pickle=True)
val_conn = np.load(val_adj_data_name, allow_pickle=True)
val_loc = np.load(val_loc_data_name, allow_pickle=True)


assert (attr_init.shape[0] == conn_init.shape[0])
data_length = attr_init.shape[0]
criterion = torch.nn.MSELoss()

train_losses = []
val_losses = []
best_loss = 100

for epoch in tqdm(range(args.epoch)):
    perm = np.random.permutation(data_length)
    attr = attr_init[perm]
    conn = conn_init[perm]
    if args.pred:
        loc = loc_init[perm]
    else:
        loc = np.zeros([data_length, 16])  # dummy padding

    # Training process
    cur_loader_idx = 0
    batch_size = args.batch_size
    train_loss = 0
    model.train()
    # loop through the dataset by batch
    while cur_loader_idx < data_length:
        cur_attr = attr[cur_loader_idx: cur_loader_idx + batch_size]
        cur_conn = conn[cur_loader_idx: cur_loader_idx + batch_size]

        cur_loc = loc[cur_loader_idx: cur_loader_idx + batch_size]
        cur_loader_idx += batch_size

        batch = tensorize(cur_attr, cur_conn)

        loc_batch = torch.tensor(cur_loc, dtype=torch.float32).cuda()

        total_step += 1

        model.zero_grad()
        dis = model(batch)
        loss = criterion(dis, loc_batch)
        train_loss += (loss.item()*len(cur_loc))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

        optimizer.step()
    

    # Validation
    cur_loader_idx = 0
    batch_size = args.batch_size
    val_loss = 0
    model.eval()
    # loop through the dataset by batch
    while cur_loader_idx < len(val_attr):
        cur_attr = val_attr[cur_loader_idx: cur_loader_idx + batch_size]
        cur_conn = val_conn[cur_loader_idx: cur_loader_idx + batch_size]

        cur_loc = val_loc[cur_loader_idx: cur_loader_idx + batch_size]
        cur_loader_idx += batch_size

        batch = tensorize(cur_attr, cur_conn)
        loc_batch = torch.tensor(cur_loc, dtype=torch.float32).cuda()

        total_step += 1

        dis = model(batch)
        loss = criterion(dis, loc_batch)
        val_loss += (loss.item()*len(cur_loc))
    
    schedler_indicator = val_loss/len(val_attr)
    scheduler.step(schedler_indicator)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if best_loss > val_loss/len(val_attr):
        best_loss = val_loss/len(val_attr)
        print("The best loss", best_loss)
        best_model = copy.deepcopy(model)
        path_to_save = os.path.join(args.save_dir,"best_model.pth")
        checkpoint = {
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, path_to_save)

    

    print("train loss", train_loss/data_length, "val loss", val_loss/len(val_attr))   
    train_losses.append(train_loss/data_length)
    val_losses.append(val_loss/len(val_attr))    


pred_list = []
gt_list = []
cur_loader_idx = 0
# Define a dummy environment for generating the images.
task_name = 'FlatTerrainTask'
task_class = getattr(tasks, "FlatTerrainTask")
FlatTerrainTask = task_class(episode_len=256)
task = FlatTerrainTask
batch_size = 32
design_idx = 0

fig_root_path = "gnn_pred_design"
if not os.path.exists(os.path.join(args.save_dir, fig_root_path)):
    os.mkdir(os.path.join(args.save_dir, fig_root_path))

while cur_loader_idx < len(val_attr):
    cur_attr = val_attr[cur_loader_idx: cur_loader_idx + batch_size]
    cur_conn = val_conn[cur_loader_idx: cur_loader_idx + batch_size]

    cur_loc = val_loc[cur_loader_idx: cur_loader_idx + batch_size]
    cur_loader_idx += batch_size

    batch = tensorize(cur_attr, cur_conn)
    loc_batch = torch.tensor(cur_loc, dtype=torch.float32).cuda()

    total_step += 1

    dis = best_model(batch)

    for j, (p,g) in enumerate(zip(dis, cur_loc)):
        cur_adj, cur_feat = cur_conn[j], cur_attr[j]
        # print(p.item(), g.item())
        pred_list.append(p.item())
        gt_list.append(g.item())
        robot = graph_to_robot(cur_adj, cur_feat)
        save_fig_path = os.path.join(args.save_dir, fig_root_path,"design{}_dof{}_pred{}_gt{}.png".format(design_idx, len(cur_feat), "{:.0f}".format(p*100),"{:.0f}".format(g*100)))
        plt.imsave(save_fig_path, get_robot_image(robot, task))
        design_idx += 1

axis = np.arange(len(pred_list))

sorted_indices = np.argsort(pred_list)
sorted_another_list = [gt_list[i] for i in sorted_indices]
for i, (p,g) in enumerate(zip(sorted(pred_list), sorted_another_list)):
    plt.plot([i,i], [p, g])

plt.scatter(axis, sorted(pred_list), c='b', label="pred value")
plt.scatter(axis, sorted_another_list, c='r', label="ground truth value")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.ylabel('distance')
plt.xlabel('data ID')
plt.show()
plt.close()


train_losses_mod = [loss*100 for loss in train_losses]
val_losses_mod = [loss*100 for loss in val_losses]
plt.plot(range(len(train_losses_mod)), train_losses_mod, label="train loss", color='r')
plt.plot(range(len(val_losses_mod)), val_losses_mod, label="val loss", color='b')   

plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")


plt.show()  
plt.close()
plt.plot(range(len(train_losses)), train_losses, label="train loss", color='r')
plt.plot(range(len(val_losses)), val_losses, label="val loss", color='b')   

plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

plt.show()  