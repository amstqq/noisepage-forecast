import torch
import os
import numpy as np


class ParamDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, class_index_arr, scaling_factors):
        # X: (N, window_size, num_quantile + num_cov)
        # Y: (N, window_size, num_quantile)
        # class_index_arr: (N, )
        # scaling_factors: (N, 2)
        self.X = X
        self.Y = Y
        self.class_index_arr = class_index_arr
        self.scaling_factors = scaling_factors
        self.len = self.X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Return format
        # input_feat, class_index, label
        return self.X[idx], int(self.class_index_arr[idx]), self.scaling_factors[idx], self.Y[idx]
