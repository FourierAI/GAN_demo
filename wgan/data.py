from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch


class COSDataset(Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    # 返回数据集大小
    def __len__(self):
        return len(self.Data)

    # 得到数据内容和标签
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor(self.Label[index])
        return data, label


class EcgDataset(Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self, samples, ecg_dim):
        self.samples = samples
        self.ecg_dim = ecg_dim
        PAINT_POINTS = np.vstack([np.linspace(-10, 10, self.ecg_dim) for _ in range(self.samples)])

        # 生成一元二次方程
        a = np.random.uniform(1, 3, size=self.samples)[:, np.newaxis]
        # f(x) = ax*2+a-1
        # a uniform \in [1,2]
        paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
        quadratic = torch.from_numpy(paintings).float()

        # 生成一元一次方程
        a = np.random.uniform(2, 6, size=self.samples)[:, np.newaxis]
        paintings = a * PAINT_POINTS + (a - 1)
        linear = torch.from_numpy(paintings).float()


        # 生成正弦信号f(x) = a*sin(X)+(a-1)
        a = np.random.uniform(1, 8, size=self.samples)[:, np.newaxis]
        paintings = a * np.sin(PAINT_POINTS) + a - 1
        sin_wave = torch.from_numpy(paintings).float()
        self.paintings = torch.cat((quadratic, linear, sin_wave), 0)

        # stack condition
        # todo resize ht dim
        sin_wave_condition = torch.ones(self.samples) * 2
        quadratic_condition = torch.ones(self.samples) * 1
        sin_wave_condition = sin_wave_condition.view(-1, 1)
        quadratic_condition = quadratic_condition.view(-1, 1)
        linear_condition = torch.zeros(self.samples).view(-1, 1)
        self.condition = torch.cat((quadratic_condition, linear_condition, sin_wave_condition), 0)

    # 返回数据集大小
    def __len__(self):
        return len(self.paintings)

    # 得到数据内容和标签
    def __getitem__(self, index):
        paintings = torch.Tensor(self.paintings[index])
        condition = torch.Tensor(self.condition[index])
        return paintings, condition


if __name__ == '__main__':

    data = EcgDataset(60000, 100)
    print(data[0])

    dataloader = DataLoader(data, shuffle=True, batch_size=2)
    for paintings, condition in dataloader:
        print(f'paintings:{paintings}\n '
              f'condition:{condition}')
