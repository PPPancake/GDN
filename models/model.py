import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class GDNLayer(nn.Module):
	def __init__(self, K, num_classes, embed_dim, inter1):
		super(GDNLayer, self).__init__()
		self.inter1 = inter1

		self.xent = nn.CrossEntropyLoss()
		#self.embed_dim = embed_dim # MLP输出维度
		self.softmax = nn.Softmax(dim=-1)
		self.KLDiv = nn.KLDivLoss(reduction='batchmean') # KL 散度损失函数, 衡量两个概率分布之间的差异
		self.cos = nn.CosineSimilarity(dim=1, eps=1e-6) 
		#self.weight = nn.Parameter(torch.FloatTensor((int(math.pow(2, K+1)-1) * self.embed_dim), self.embed_dim))
		#self.weight = nn.Parameter(torch.FloatTensor((2 * self.embed_dim), self.embed_dim))git 
		self.weight = nn.Parameter(torch.FloatTensor(96, inter1.embed_dim))
		self.weight2 = nn.Parameter(torch.FloatTensor(inter1.embed_dim, num_classes))
		#self.weight_mlp = nn.Parameter(torch.FloatTensor(self.embed_dim, num_classes))
		self.fn = nn.LeakyReLU(0.3)
		
		init.xavier_uniform_(self.weight) # 初始化weight参数
		init.xavier_uniform_(self.weight2)
		#init.xavier_uniform_(self.weight_mlp)

	# 前向传播
	def forward(self, nodes, labels):
		embeds1 = self.inter1(nodes, labels)

		# scores = embeds1.mm(self.weight)

		scores = embeds1.mm(self.weight)
		scores = self.fn(scores)
		scores = scores.mm(self.weight2)

		#scores_mlp = embeds1[:, 0: self.embed_dim].mm(self.weight_mlp)
		#scores_mlp = self.fn(scores_mlp) # 送入多层感知机		

		return scores
	
	# 将模型的输出通过softmax转为概率
	def to_prob(self, nodes, labels):
		gnn_logits = self.forward(nodes, labels)
		gnn_scores = self.softmax(gnn_logits)
		return gnn_scores
	
	# 交叉熵损失函数
	def loss(self, nodes, labels):
		gnn_scores = self.forward(nodes, labels)
		gnn_loss = self.xent(gnn_scores, labels.squeeze())
		return gnn_loss

