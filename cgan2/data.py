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


class PaintingDataset(Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self):
        SAMPLES = 64 * 10000
        ART_COMPONENTS = 50
        PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(SAMPLES)])

        # 生成一元二次方程
        a = np.random.uniform(1, 2, size=SAMPLES)[:, np.newaxis]
        # f(x) = ax*2+a-1
        # a uniform \in [1,2]
        paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
        quadratic = torch.from_numpy(paintings).float()

        # 生成一元一次方程
        a = np.random.uniform(1, 2, size=SAMPLES)[:, np.newaxis]
        paintings = a * PAINT_POINTS + (a - 1)
        linear = torch.from_numpy(paintings).float()
        self.paintings = torch.cat((quadratic, linear), 0)

        # stack condition
        quadratic_condition = torch.ones(SAMPLES) * 2
        quadratic_condition = quadratic_condition.view(-1, 1)
        linear_condition = torch.ones(SAMPLES).view(-1, 1)
        self.condition = torch.cat((quadratic_condition, linear_condition), 0)

    # 返回数据集大小
    def __len__(self):
        return len(self.paintings)

    # 得到数据内容和标签
    def __getitem__(self, index):
        paintings = torch.Tensor(self.paintings[index])
        condition = torch.Tensor(self.condition[index])
        return paintings, condition


if __name__ == '__main__':

    data = PaintingDataset()
    print(data[0])

    dataloader = DataLoader(data, shuffle=True, batch_size=2)
    for paintings, condition in dataloader:
        print(f'paintings:{paintings}\n '
              f'condition:{condition}')
