import torch
import torch.nn as nn
import torch.nn.functional as F
import models.AGCN as AGCN


class MyModel(nn.Module):
    def __init__(self, batch_size, hidden_size=32, time_steps=50, feature=4, num_car=7):
        """
        :param batch_size:
        :param hidden_size:
        :param time_steps:
        :param feature:
        :param num_car:
        """
        super(MyModel, self).__init__()
        self.AGC = AGCN.AdaptiveGraphConv(hidden_size=hidden_size)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.time_steps = time_steps
        self.features = feature
        self.num_car = num_car
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, batch_first=True)  # position_x
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)  # position_y
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)  # speed_x
        self.lstm4 = nn.LSTM(hidden_size, hidden_size, batch_first=True)  # speed_y

        # 输出网络
        self.output_layer = nn.Linear(self.hidden_size * 4 * self.num_car, 1 * 3)

    def forward(self, GCNinput):
        """
        计算前向转播
        :param GCNinput: (batch, time, num_area, feature_
        :return: flow, empty, demand
        """
        # GCNinput shape torch.Size([64, 20, 100, 10])
        # ## 输入GCN
        GCNoutput = self.AGC.forward(GCNinput)
        # output: batch, time, num_area, hidden_size ? hidden_size ?

        # ##输入LSTM
        # 1.time和num_area位置交换 (batch, num_area, time, feature) -> (batch, time, num_area, feature)
        # 2.batch和num_area合并 (batch, time, num_area, feature) -> (batch*time, num_area, feature)

        # GCNoutput shape torch.Size([64, 20, 100, 64])
        GCNoutput = GCNoutput.reshape(self.batch_size * self.num_car, self.time_steps, self.hidden_size)
        # 3.输入Lstm网络
        GCNoutput = F.sigmoid(GCNoutput)
        Lstm_output_positionX, _ = self.lstm1(GCNoutput)
        Lstm_output_positionY, _ = self.lstm2(GCNoutput)
        Lstm_output_speedX, _ = self.lstm3(GCNoutput)
        Lstm_output_speedY, _ = self.lstm4(GCNoutput)

        # 输出 LSTM_output_*: batch_size * num_area, time, hidden_size

        # 4.抽取时间维上取最后一个数据
        Lstm_output_positionX = Lstm_output_positionX[:, -1, :].squeeze()
        Lstm_output_positionY = Lstm_output_positionY[:, -1, :].squeeze()
        Lstm_output_speedX = Lstm_output_speedX[:, -1, :].squeeze()
        Lstm_output_speedY = Lstm_output_speedY[:, -1, :].squeeze()

        # 6.构造输出[0, 1, 0]
        # 首先拼接时空融合结果
        splicing = torch.cat([Lstm_output_positionX, Lstm_output_positionY, Lstm_output_speedX, Lstm_output_speedY], dim=-1)

        splicing = splicing.reshape(self.batch_size, -1)
        output = self.output_layer(splicing)
        output = F.sigmoid(output)
        return output
