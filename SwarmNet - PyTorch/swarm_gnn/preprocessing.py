# Readies a frame for training/testing
import copy

import numpy
import pandas

# Returns a data frame containing only data for specific agent IDs
from sklearn.preprocessing import MinMaxScaler


def preprocess(data):
    # TODO this is a test function using existing data. Throws away all frames without equal number of agents.
    data_list = []
    steps = data["Time"].unique()
    ids = data["ID"].unique()
    for step in steps:
        step_data = []
        step_frame = data[(data["Time"] == step)]
        agent_ids = step_frame["ID"].unique()
        if len(agent_ids) != len(ids):
            continue
        for agent_id in agent_ids:
            agent_frame = step_frame[(step_frame["ID"] == agent_id)]
            agent_x = agent_frame["X Global"].unique()[0]
            agent_y = agent_frame["Y Global"].unique()[0]
            agent_r = agent_frame["Velocity R"].unique()[0]
            agent_theta = agent_frame["Velocity Theta"].unique()[0]
            coord = [agent_x, agent_y, agent_r, agent_theta]
            step_data.append(coord)
        data_list.append(step_data)
    data_list = numpy.array(data_list, dtype=float)

    return data_list
