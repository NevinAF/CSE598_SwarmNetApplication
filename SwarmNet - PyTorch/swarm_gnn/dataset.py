import json
import math
import os

import numpy
import pandas
import torch
from torch.utils.data import Dataset

from swarm_gnn.preprocessing import preprocess


# Retrieve data
def retrieve_dataset(config, scaler):
    train_dataset = SimulationDataset(config.train_path, False, scaler, config)
    test_dataset = None

    return train_dataset, test_dataset


class SimulationDataset(Dataset):

    def __init__(self, path, testData=False, scaler=None,
                 config=None, mode=1):
        super().__init__()
        # data = numpy.array([[[0, 1], [9, 9], [8, 7]],
        #                     [[1, 2], [10, 10], [7, 6]],
        #                     [[2, 3], [11, 11], [6, 5]],
        #                     [[3, 4], [12, 12], [5, 4]],
        #                     [[4, 5], [13, 13], [4, 3]],
        #                     [[5, 6], [14, 14], [3, 2]],
        #                     [[5, 6], [15, 15], [2, 1]],
        #                     [[6, 7], [16, 16], [1, 0]],
        #                     [[7, 8], [17, 17], [0, -1]],
        #                     [[9, 10], [19, 19], [-2, -3]],
        #                     [[11, 12], [21, 21], [-4, -5]]
        #                     ], dtype=float)
        prediction_steps = config.prediction_steps
        data = pandas.read_csv(path)
        data = preprocess(data)
        # Data reshaped to  [time_step, agent, state]
        data = numpy.swapaxes(data, 0, 1)
        if config.truth_available:
            truth_ends_at = data.shape[1] - prediction_steps + 1
            # Ground truth starts at 7 time-steps # TODO should be variable based on num layers and kernel size
            self.data_y = data[:, 7:truth_ends_at, :]
            # Don't want to predict on time-steps where truth no longer available
            self.data_x = data[:, 0:truth_ends_at - 1, :]
        else:
            truth_ends_at = None
        self.state_length = data.shape[2]
        # Scaler incorporated but probably will not be used for some time
        self.scaler = scaler
        # If using scikit scaler, scale data (training data uses specified scaler, testing data uses training scaler)
        self.X = torch.tensor(self.data_x, requires_grad=True)
        self.y = torch.tensor(self.data_y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
