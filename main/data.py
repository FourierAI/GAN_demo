from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import os



class BCG2ECGDataset(Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self):
        self.bcgs, self.ecgs = self.__loaddata()

    def __loaddata(self):
        bcgs = []
        ecgs = []
        bcg_dir_path = '../bcg_ecg/BCG_ECG/datasets/BCG_h(t)'
        ecg_dir_path = '../bcg_ecg/BCG_ECG/datasets/ECG_h(t)'
        bgs_dirs = os.listdir(bcg_dir_path)
        for file_name in bgs_dirs:
            bcg_full_path = os.path.join(bcg_dir_path, file_name)
            with open(bcg_full_path) as file:
                for line in file:
                    bcg = [float(i) for i in line.split(',')]
                    bcgs.append(bcg)

            ecg_full_path = os.path.join(ecg_dir_path, file_name)
            with open(ecg_full_path) as file:
                for line in file:
                    ecg = [float(i) for i in line.split(',')]
                    ecgs.append(ecg)
        return bcgs, ecgs

    # 返回数据集大小
    def __len__(self):
        return len(self.bcgs)

    # 得到数据内容和标签
    def __getitem__(self, index):
        bcgs = torch.Tensor(self.bcgs[index])
        ecgs = torch.Tensor(self.ecgs[index])
        return bcgs, ecgs

if __name__ == '__main__':

    # data = BCG2ECGDataset()
    # print(data[0])
    #
    # dataloader = DataLoader(data, shuffle=True, batch_size=2)
    # for bcgs, ecgs in dataloader:
    #     print(f'bcgs:{bcgs}\n '
    #           f'ecgs:{ecgs}')

    path = '../bcg_ecg/BCG_ECG/datasets/BCG_h(t)'
    bgs_dirs = os.listdir(path)
    for file_name in bgs_dirs:
        full_path = os.path.join(path, file_name)
        print(full_path)