from __future__ import print_function, division

import torch
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")
plt.ion()

class Route_data(Dataset):
    """Route_data dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.data_frame.iloc[idx, 0])
        # train_data= self.data_frame.iloc[idx, 1:11]
        train_data = self.data_frame.iloc[idx, 1:9]
        train_data = np.array([train_data])
        train_data= train_data.astype('float').reshape(-1, 1)
        # label = self.data_frame.iloc[idx, 11]
        label = self.data_frame.iloc[idx, 9]
        label = np.array([label])
        label = label.astype('float').reshape(-1, 1)
        sample = {'train_data': train_data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

#数据分批处理，每一批大小是一个batch
# my_dataset = Route_data(csv_file='TTC.csv',
#                                     root_dir='.')
#
#
# dataloader = DataLoader(my_dataset, batch_size=4,
#                         shuffle=True)
# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['train_data'],
#           sample_batched['label'])