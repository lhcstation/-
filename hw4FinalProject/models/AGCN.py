import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphConv(nn.Module):
    def __init__(self, hidden_size, feature_dim=4, node_num=7, embedding_size=128):
        """
        :param hidden_size: 隐层个数
        :param feature_dim: 参数维度
        :param node_num: 节点个数
        :param embedding_size: 节点嵌入的维度
        """
        super(AdaptiveGraphConv, self).__init__()
        self.embedding_spatial = nn.Parameter(torch.Tensor(node_num, embedding_size))

        self.conv_spatial = nn.Linear(feature_dim, hidden_size)  # wx+b
        # nn.Linear(输入的二维张量的大小, 输出的二维张量的大小)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)  # 初始化——Xavier均匀分布:通过网络层时，输入和输出的方差相同

    def forward(self, x):
        # x: batch, time, num_car, feature
        adj_spatial = torch.eye(self.embedding_spatial.shape[0]).to(self.embedding_spatial.device) + \
                      F.softmax(F.relu(torch.mm(self.embedding_spatial,
                                                self.embedding_spatial.transpose(0, 1))), dim=1)
        # 计算邻接矩阵

        # batch * time, num_car, feature
        # 为了计算矩阵乘法的方便
        feature_spatial = x.reshape(x.shape[0] * x.shape[1], x.shape[2], -1).to(
            self.embedding_spatial.device)  # 将四维变成三维  batch 与 time 合并

        adj_spatial = torch.tensor(adj_spatial, dtype=torch.float)
        feature_spatial = torch.tensor(feature_spatial, dtype=torch.float)
        spatial_res = self.conv_spatial(torch.matmul(adj_spatial, feature_spatial))  # AXW+B
        spatial_res = spatial_res.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        # spatial_res = torch.softmax(spatial_res)

        return spatial_res
