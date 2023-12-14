from torch.autograd import Variable
import torch
# to_neighs_list = [[2,3], [], [4], [3,5], [4]]
# samp_neighs = [set(x) for x in to_neighs_list]

# unique_nodes_list = list(set.union(*samp_neighs))
# unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
# print(unique_nodes_list)
# print(unique_nodes)

# mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
# column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
# row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
# mask[row_indices, column_indices] = 1 # 对值为1的节点进行采样
# print(mask)
# num_neigh = mask.sum(1, keepdim=True)
# print(num_neigh)
# mask = mask.div(num_neigh)
# print(mask)

# self_feats = torch.tensor([[1,2], [2, 0], [2, 4]])
# agg_feats = torch.tensor([[2], [2], [3]])
# cat_feats = torch.cat((self_feats, agg_feats), dim=1)
# print(cat_feats)

labels = torch.tensor([[1, 2],[2, 3],[4, 5]])

print(labels.squeeze())