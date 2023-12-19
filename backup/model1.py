import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
"""
	GDN Model
"""


class GDNLayer(nn.Module):
	"""
	One GDN layer
	"""

	def __init__(self, num_classes, inter1):
		"""
		Initialize GDN model
		:param num_classes: number of classes (2 in our paper)
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
		super(GDNLayer, self).__init__()
		self.inter1 = inter1
		self.xent = nn.CrossEntropyLoss() # 交叉熵损失函数
		self.softmax = nn.Softmax(dim=-1) 
		# 创建一个 Kullback-Leibler Divergence 损失函数，该函数在计算概率分布之间的差异时使用
		self.KLDiv = nn.KLDivLoss(reduction='batchmean')
		# 余弦相似度
		self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
		# the parameter to transform the final embedding
		# 可学习的参数weight，其维度为 (num_classes, inter1.embed_dim)，用于将最终的嵌入进行线性变换
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter1.embed_dim))
		# 使用 Xavier 初始化方法对权重参数进行初始化。 Xavier 初始化有助于确保网络在初始阶段的稳定性
		init.xavier_uniform_(self.weight)

	
	# 前向传播
	def forward(self, nodes, labels):
		# 使用传入的 inter-relation aggregator 计算节点的嵌入
		embeds1 = self.inter1(nodes, labels)
		# 使用参数权重对嵌入进行变换，得到模型的输出分数
		scores = self.weight.mm(embeds1)
		return scores.t()

	# 将模型的输出通过softmax转为概率
	def to_prob(self, nodes, labels):
		gnn_logits = self.forward(nodes, labels)
		gnn_scores = self.softmax(gnn_logits)
		return gnn_scores

	# 交叉熵损失函数
	def loss(self, nodes, labels):
		gnn_scores = self.forward(nodes, labels)
		# GNN loss
		gnn_loss = self.xent(gnn_scores, labels.squeeze())
		return gnn_loss
	