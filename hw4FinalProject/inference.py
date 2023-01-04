import torch
import numpy as np
from utils.dataloader import MyDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


def ReshapeLabel(sequence):
    s = sequence.detach().numpy()
    s = np.int64(s > 0)
    return s


highRiskF1, lowRishF1, laneKeepF1 = [], [], []
def test(model, testLoader, epoch):
    model.eval()
    for i in range(epoch):
        for step, data in enumerate(testLoader):
            testX, testY = data[0].to(torch.device('cuda')), data[1].to(torch.device('cuda'))
            predY = model(testX).to(torch.device('cpu'))
            label = ReshapeLabel(predY)
            testY = testY.to(torch.device('cpu')).detach().numpy()
            highRiskF1.append(f1_score(testY[:, 0], label[:, 0]))
            lowRishF1.append(f1_score(testY[:, 1], label[:, 1]))
            laneKeepF1.append(f1_score(testY[:, 2], label[:, 2]))

if __name__ == '__main__':

    testHR = './data/test_highRisk.npy'
    testLR = './data/test_lowRisk.npy'
    testLC = './data/test_laneKeep.npy'

    testData = MyDataset(testHR, testLR, testLC)
    testLoader = DataLoader(dataset=testData, batch_size=64, shuffle=True, drop_last=True)
    model = torch.load('./result/model.pkl')

    test(model, testLoader, 1)
    print("==================== 测试集上模型表现 ====================")
    print("highRiskF1: ", sum(highRiskF1)/len(highRiskF1))
    print("lowRishF1: ", sum(lowRishF1) / len(lowRishF1))
    print("laneKeepF1: ", sum(laneKeepF1) / len(laneKeepF1))

