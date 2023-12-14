import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class GDNLayer(nn.Module):
    def __init__(self, num_classes, inter1):
        super(GDNLayer, self).__init__()
        self.inter1 = inter1
        self.xent = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        self.KLDiv = nn.KLDivLoss(reduction='batchmean') # KL 散度损失函数, 衡量两个概率分布之间的差异
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6) # 
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter1.embed_dim))
        init.xavier_uniform_(self.weight) # 初始化weight参数

    # 前向传播
    def forward(self, nodes, labels):
        embeds1 = self.inter1(nodes, labels)
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
        gnn_loss = self.xent(gnn_scores, labels.squeeze())
        return gnn_loss
