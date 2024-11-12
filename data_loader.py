from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch


class iris_dataloader(Dataset):
    # 重写三种方法

    def __init__(self, data_path):
        self.data_path = data_path
        assert os.path.exists(self.data_path), "data does not exist"

        df = pd.read_csv(self.data_path, names=[0, 1, 2, 3, 4])

        mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

        df[4] = df[4].map(mapping)

        data = df.iloc[:, :4]
        label = df.iloc[:, 4:]

        # 数据归一化 mean 0 方差为1 Z值化
        data = (data - np.mean(data) / np.std(data))

        self.data = torch.from_numpy(np.array(data, dtype="float32"))
        self.label = torch.from_numpy(np.array(label, dtype="float32"))

        self.data_num = len(label)
        print("当前数据集的大小：", self.data_num)

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        self.data = list(self.data)
        self.label = list(self.label)

        return self.data[index], self.label[index]
