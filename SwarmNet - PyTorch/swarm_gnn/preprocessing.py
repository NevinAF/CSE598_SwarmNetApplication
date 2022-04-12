# Readies a frame for training/testing
import copy

import numpy
import pandas


# Returns a data frame containing only data for specific agent IDs
# from sklearn.preprocessing import MinMaxScaler


def preprocess_csv(data):
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


def preprocess_json(data):
    data = numpy.array(data)
    # TODO temporarily ignoring environmental context and fitting to exact environment
    data = data[:, :, 0:6]

    return data


def preprocess_predict_steps(data, is_test_data, steps, truth_available):
    if is_test_data is True:
        data = data[:, :500, ]
    if truth_available:
        truth_ends_at = data.shape[1] - steps + 1
        # Ground truth starts at 7 time-steps # TODO should be variable based on num layers and kernel size
        # Shape = [agent, condensed time-step, prediction step, state-vector]
        # Don't want to predict on time-steps where truth no longer available
        data_x = data[:, 0:truth_ends_at - 1, :]
        data_y = numpy.zeros([data.shape[0], data_x.shape[1], steps, data.shape[2]])
        for i in range(0, steps):
            data_y[:, 6:, i, :] = data[:, 7 + i:truth_ends_at + i, :]
        data_x = numpy.swapaxes(data_x, 0, 1)
        # Shape [step, agent, predicted step, state] e.g. [0][1][0][0] is the 1-step x prediction for agent 1 at \
        #   time-step 0 (6)
        data_y = numpy.swapaxes(data_y, 0, 1)
    else:
        truth_ends_at = None
    state_length = data.shape[2]
    return data_x, data_y, state_length
