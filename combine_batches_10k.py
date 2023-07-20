import numpy as np
import os
import random 

root_path1 = "0to2k_data/save_dir_withdis_2k_flat"
adj_batch1 = np.load(os.path.join(root_path1, "0to2k_adj.npy"), allow_pickle=True)
feat_batch1 = np.load(os.path.join(root_path1, "0to2k_feat.npy"), allow_pickle=True)
dis_batch1 = np.load(os.path.join(root_path1, "0to2k_dis.npy"), allow_pickle=True)
print(len(adj_batch1))

root_path2 = "2kto4k_data/save_dir_withdis_2k_flat"
adj_batch2 = np.load(os.path.join(root_path2, "2kto4k_adj.npy"), allow_pickle=True)
feat_batch2 = np.load(os.path.join(root_path2, "2kto4k_feat.npy"), allow_pickle=True)
dis_batch2 = np.load(os.path.join(root_path2, "2kto4k_dis.npy"), allow_pickle=True)
print(len(adj_batch2))

root_path3 = "4kto5k_data/save_dir_withdis_2k_flat"
adj_batch3 = np.load(os.path.join(root_path3, "4kto5k_adj.npy"), allow_pickle=True)
feat_batch3 = np.load(os.path.join(root_path3, "4kto5k_feat.npy"), allow_pickle=True)
dis_batch3 = np.load(os.path.join(root_path3, "4kto5k_dis.npy"), allow_pickle=True)
print(len(adj_batch3))

root_path4 = "5kto6k_data/save_dir_withdis_2k_flat"
adj_batch4 = np.load(os.path.join(root_path4, "5kto6k_adj.npy"), allow_pickle=True)
feat_batch4 = np.load(os.path.join(root_path4, "5kto6k_feat.npy"), allow_pickle=True)
dis_batch4 = np.load(os.path.join(root_path4, "5kto6k_dis.npy"), allow_pickle=True)
print(len(adj_batch4))

root_path5 = "6kto7k_data/save_dir_withdis_2k_flat"
adj_batch5 = np.load(os.path.join(root_path5, "6kto7k_adj.npy"), allow_pickle=True)
feat_batch5 = np.load(os.path.join(root_path5, "6kto7k_feat.npy"), allow_pickle=True)
dis_batch5 = np.load(os.path.join(root_path5, "6kto7k_dis.npy"), allow_pickle=True)
print(len(adj_batch5))

root_path6 = "7kto8k_data/save_dir_withdis_2k_flat"
adj_batch6 = np.load(os.path.join(root_path6, "7kto8k_adj.npy"), allow_pickle=True)
feat_batch6 = np.load(os.path.join(root_path6, "7kto8k_feat.npy"), allow_pickle=True)
dis_batch6 = np.load(os.path.join(root_path6, "7kto8k_dis.npy"), allow_pickle=True)
print(len(adj_batch6))

root_path7 = "8kto9k_data/save_dir_withdis_2k_flat"
adj_batch7 = np.load(os.path.join(root_path7, "8kto9k_adj.npy"), allow_pickle=True)
feat_batch7 = np.load(os.path.join(root_path7, "8kto9k_feat.npy"), allow_pickle=True)
dis_batch7 = np.load(os.path.join(root_path7, "8kto9k_dis.npy"), allow_pickle=True)
print(len(adj_batch7))

root_path8 = "9kto10k_data/save_dir_withdis_2k_flat"
adj_batch8 = np.load(os.path.join(root_path8, "9kto10k_adj.npy"), allow_pickle=True)
feat_batch8 = np.load(os.path.join(root_path8, "9kto10k_feat.npy"), allow_pickle=True)
dis_batch8 = np.load(os.path.join(root_path8, "9kto10k_dis.npy"), allow_pickle=True)
print(len(adj_batch8))

def keep_leading_zero(matrix):
    result = np.zeros_like(matrix)
    for i, row in enumerate(matrix):
        leading_zero = np.argmax(row == 1)
        result[i, leading_zero] = 1
    return result

combine_adj = np.concatenate((adj_batch1,adj_batch2,adj_batch3,adj_batch4,
                              adj_batch5,adj_batch6,adj_batch7,adj_batch8))
combine_feat = np.concatenate((feat_batch1,feat_batch2,feat_batch3,feat_batch4,
                               feat_batch5,feat_batch6,feat_batch7,feat_batch8))
combine_dis = np.concatenate((dis_batch1,dis_batch2,dis_batch3,dis_batch4,
                              dis_batch5,dis_batch6,dis_batch7,dis_batch8))


train_adj = []
train_feat = []
train_dis = []
val_adj = []
val_feat = []
val_dis = []


# dic = {}
# for i, feat in enumerate(combine_feat[:3600]):
#     dof = len(feat)
#     if dof == 11:
#         train_adj.append(combine_adj[i])
#         train_feat.append(feat)
#         train_dis.append(combine_dis[i])

#     if dof in dic:
#         dic[dof] += 1
#     else:
#         dic[dof] = 1

# dic_val = {}

# for i, feat in enumerate(combine_feat[3600:]):
#     # print(i+3600)
#     dof = len(feat)
#     if dof == 11:
#         val_adj.append(combine_adj[i+3600])
#         val_feat.append(feat)
#         val_dis.append(combine_dis[i+3600])

#     if dof in dic_val:
#         dic_val[dof] += 1
#     else:
#         dic_val[dof] = 1
# print(dic)
# print(dic_val)
# print("type", type(val_adj))
# Iterate through the batches, count the number of data in each dof
adj_count = {}
feat_count = {}
dis_count = {}
for i, (adj, feat, dis) in enumerate(zip(combine_adj, combine_feat, combine_dis)):
    dof = len(feat)

    if dof in adj_count:
        adj_count[dof].append(adj)
        feat_count[dof].append(feat)
        dis_count[dof].append(dis)
    else:
        adj_count[dof] = [adj]
        feat_count[dof] = [feat]
        dis_count[dof] = [dis]

train_adj = []
train_feat = []
train_dis = []
val_adj = []
val_feat = []
val_dis = []
train_ratio = 0.9 
for key in adj_count.keys():
    train_idx = int(len(adj_count[key])*train_ratio)
    print(train_idx, len(adj_count[key]), len(adj_count[key][:train_idx]))
    # train_adj = train_adj + adj_count[key][:train_idx]
    # train_feat = train_feat + feat_count[key][:train_idx]
    # train_dis = train_dis + dis_count[key][:train_idx]
    for adj, feat, dis in zip(adj_count[key][:train_idx], \
                              feat_count[key][:train_idx],\
                              dis_count[key][:train_idx]):
        # train_adj.append(adj)
        train_adj.append(keep_leading_zero(adj))
        train_feat.append(feat)
        train_dis.append(dis)
    # val_adj = val_adj + adj_count[key][train_idx:]
    # val_feat = val_feat + feat_count[key][train_idx:]
    # val_dis = val_dis + dis_count[key][train_idx:]
    for adj, feat, dis in zip(adj_count[key][train_idx:], \
                              feat_count[key][train_idx:],\
                              dis_count[key][train_idx:]):
        # val_adj.append(adj)
        print(adj, keep_leading_zero(adj))
        val_adj.append(keep_leading_zero(adj))
        val_feat.append(feat)
        val_dis.append(dis)

# Shuffle the list
combined_list = list(zip(train_adj, train_feat, train_dis))
random.shuffle(combined_list)
train_adj, train_feat, train_dis = zip(*combined_list)

# Shuffle the list
combined_list = list(zip(val_adj, val_feat, val_dis))
random.shuffle(combined_list)
val_adj, val_feat, val_dis = zip(*combined_list)
print(type(val_adj), len(val_adj))


# for adj in train_adj:
#     print(adj.shape)
#     if adj.shape[0] != adj.shape[1]:
#         print("break")
#         break
# combined_lists = list(zip(val_adj, val_feat, val_dis))
# val_adj, val_feat, val_dis = zip(*combined_lists)

print("Final shape", len(train_adj), len(val_adj))
train_adj_np = np.empty(len(train_adj), dtype=object)
train_adj_np[:] = train_adj
# float_array = np.empty(train_adj_np.shape, dtype=np.float32)
# for i, arr in enumerate(train_adj_np):
#     float_array[i] = arr.astype(np.float32)
train_feat_np = np.empty(len(train_feat), dtype=object)
train_feat_np[:] = train_feat
train_dis_np = np.empty(len(train_dis), dtype=object)
train_dis_np[:] = train_dis

val_adj_np = np.empty(len(val_adj), dtype=object)
val_adj_np[:] = val_adj
val_feat_np = np.empty(len(val_feat), dtype=object)
val_feat_np[:] = val_feat
val_dis_np = np.empty(len(val_dis), dtype=object)
val_dis_np[:] = val_dis
save_dir = "data/new_train_data_dis_10k"

np.save(os.path.join(save_dir, "train_adj_even_kl.npy"), train_adj_np)
np.save(os.path.join(save_dir, "train_feat_even_kl.npy"), train_feat_np)
np.save(os.path.join(save_dir, "train_dis_even_kl.npy"), train_dis)
# import pickle 
# with open(os.path.join(save_dir, "train_adj_even.pkl"), "wb") as file:
#     pickle.dump(train_adj, file)
# with open(os.path.join(save_dir, "train_feat_even.pkl"), "wb") as file:
#     pickle.dump(train_feat, file)
# with open(os.path.join(save_dir, "train_dis_even.pkl"), "wb") as file:
#     pickle.dump(train_dis, file)

np.save(os.path.join(save_dir, "val_adj_even_kl.npy"), val_adj_np)
np.save(os.path.join(save_dir, "val_feat_even_kl.npy"), val_feat_np)
np.save(os.path.join(save_dir, "val_dis_even_kl.npy"), val_dis)
# with open(os.path.join(save_dir, "val_adj_even.pkl"), "wb") as file:
#     pickle.dump(val_adj, file)
# with open(os.path.join(save_dir, "val_feat_even.pkl"), "wb") as file:
#     pickle.dump(val_feat, file)
# with open(os.path.join(save_dir, "val_dis_even.pkl"), "wb") as file:
#     pickle.dump(val_dis, file)
# print(len(combine_adj), combine_dis.shape)

# save_dir = "data/new_train_data_dis_4k"
# np.save(os.path.join(save_dir, "adj.npy"), combine_adj)
# np.save(os.path.join(save_dir, "feat.npy"), combine_feat)
# np.save(os.path.join(save_dir, "dis.npy"), combine_dis)

# np.save(os.path.join(save_dir, "train_adj.npy"), combine_adj[:3600])
# np.save(os.path.join(save_dir, "train_feat.npy"), combine_feat[:3600])
# np.save(os.path.join(save_dir, "train_dis.npy"), combine_dis[:3600])

# np.save(os.path.join(save_dir, "val_adj.npy"), combine_adj[3600:])
# np.save(os.path.join(save_dir, "val_feat.npy"), combine_feat[3600:])
# np.save(os.path.join(save_dir, "val_dis.npy"), combine_dis[3600:])