import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.model import MyModel
from utils.dataloader import MyDataset
from torch.utils.data import DataLoader


def train(model, trainLoader, optimizer, epoch):
    model.train()
    for i in range(epoch):
        for step, data in enumerate(trainLoader):
            trainX, trainY = data[0].to(torch.device('cuda')), data[1].to(torch.device('cuda'))
            predY = model(trainX)
            loss = F.cross_entropy(predY, trainY)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            print('epoch:', i + 1, 'step:', step + 1, 'loss:', loss.item())


if __name__ == '__main__':
    trainHR = './data/train_highRisk.npy'
    trainLR = './data/train_lowRisk.npy'
    trainLC = './data/train_laneKeep.npy'

    model = MyModel(batch_size=64).to(torch.device('cuda'))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainData = MyDataset(trainHR, trainLR, trainLC)
    trainLoader = DataLoader(dataset=trainData, batch_size=64, shuffle=True, drop_last=True)

    print("======================== 开始训练 ========================")
    train(model, trainLoader, optimizer, 20)
    print("======================== 完成训练 ========================")

    torch.save(model, './result/model.pkl')