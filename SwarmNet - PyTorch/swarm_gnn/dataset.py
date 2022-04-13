import json
import math
import os

import numpy
import pandas
import torch
from torch.utils.data import Dataset

from swarm_gnn.preprocessing import preprocess_csv, preprocess_json, preprocess_predict_steps


# Retrieve data
def retrieve_dataset(config, scaler, predict_steps):
    train_dataset = SimulationDataset(config.train_path, False, scaler, config, predict_steps)
    test_dataset = SimulationDataset(config.test_path, True, scaler, config, predict_steps)

    return train_dataset, test_dataset


class SimulationDataset(Dataset):

    def __init__(self, path, testData=False, scaler=None,
                 config=None, predict_steps=1, mode=1):
        super().__init__()
        # self.original_data = numpy.array([[[0, 1], [9, 9], [8, 7]],
        #                                   [[1, 2], [10, 10], [7, 6]],
        #                                   [[2, 3], [11, 11], [6, 5]],
        #                                   [[3, 4], [12, 12], [5, 4]],
        #                                   [[4, 5], [13, 13], [4, 3]],
        #                                   [[5, 6], [14, 14], [3, 2]],
        #                                   [[5, 6], [15, 15], [2, 1]],
        #                                   [[6, 7], [16, 16], [1, 0]],
        #                                   [[7, 8], [17, 17], [0, -1]],
        #                                   [[9, 10], [19, 19], [-2, -3]],
        #                                   [[11, 12], [21, 21], [-4, -5]]
        #                                   ], dtype=float)
        self.prediction_steps = predict_steps
        self.original_data = self.load(path)
        self.original_data = numpy.nan_to_num(self.original_data)
        # Data reshaped to  [time_step, agent, state]
        # self.original_data = numpy.swapaxes(self.original_data, 0, 1)
        # data = data[45:]
        self.data_x, self.data_y, self.state_length = preprocess_predict_steps(self.original_data, testData,
                                                                               self.prediction_steps,
                                                                               config.truth_available)
        # Scaler incorporated but probably will not be used for some time
        self.scaler = scaler
        # If using scikit scaler, scale data (training data uses specified scaler, testing data uses training scaler)
        self.X = torch.tensor(self.data_x)
        self.y = torch.tensor(self.data_y)
        # print(self.y[0])

    def __len__(self):
        return len(self.X)-self.prediction_steps

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def load(self, path):
        data = None
        if path.endswith('.csv'):
            data = pandas.read_csv(path)
            data = preprocess_csv(data)
        elif path.endswith('.json'):
            f = open(path)
            data = json.load(f)
            data = preprocess_json(data)

        # Open with appropriate library
        # Preprocess

        return data
