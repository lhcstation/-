import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, HRfile, LRfile, LCfile):
        self.dataHR = np.load(HRfile)
        self.dataLR = np.load(LRfile)
        self.dataLC = np.load(LCfile)
        self.label, self.data = None, None

    def __getitem__(self, index):
        # 制作标签
        labelHR = [[1, 0, 0]] * self.dataHR.shape[0]
        labelLR = [[0, 1, 0]] * self.dataLR.shape[0]
        labelLC = [[0, 0, 1]] * self.dataLC.shape[0]

        self.label = np.append(labelHR, labelLR, axis=0)
        self.label = np.append(self.label, labelLC, axis=0)
        self.label = self.label.astype(float)

        self.data = np.append(self.dataHR, self.dataLR, axis=0)
        self.data = np.append(self.data, self.dataLC, axis=0)
        return self.data[index], self.label[index]

    def __len__(self):
        return self.dataHR.shape[0] + self.dataLR.shape[0] + self.dataLC.shape[0]


