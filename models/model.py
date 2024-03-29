import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class GDNLayer(nn.Module):
	def __init__(self, K, num_classes, embed_dim, inter1):
		super(GDNLayer, self).__init__()
		self.inter1 = inter1

		self.xent = nn.CrossEntropyLoss()
		self.softmax = nn.Softmax(dim=-1)
		self.KLDiv = nn.KLDivLoss(reduction='batchmean') # KL散度损失函数, 衡量两个概率分布之间的差异
		self.cos = nn.CosineSimilarity(dim=1, eps=1e-6) 
		#self.weight = nn.Parameter(torch.FloatTensor((int(math.pow(2, K+1)-1) * inter1.embed_dim), inter1.embed_dim))
		#self.weight = nn.Parameter(torch.FloatTensor((2 * self.embed_dim), self.embed_dim))
		self.weight = nn.Parameter(torch.FloatTensor(64 * 4, inter1.embed_dim))
		self.weight2 = nn.Parameter(torch.FloatTensor(inter1.embed_dim, num_classes))
		self.fn = nn.LeakyReLU(0.3)
		
		init.xavier_uniform_(self.weight) # initialize the parameter weight
		init.xavier_uniform_(self.weight2)

	def forward(self, nodes):
		embeds1 = self.inter1(nodes)

		scores = embeds1.mm(self.weight)
		# combined = F.relu(cat_feats.mm(self.weight))
		scores = self.fn(scores)
		scores = scores.mm(self.weight2)

		return scores
	
	# convert the output to probability via softmax
	def to_prob(self, nodes):
		gnn_logits = self.forward(nodes)
		gnn_scores = self.softmax(gnn_logits)
		return gnn_scores
	
	# cross Entropy Loss Function
	def loss(self, nodes, labels):
		gnn_scores = self.forward(nodes)
		gnn_loss = self.xent(gnn_scores, labels.squeeze())
		return gnn_loss

