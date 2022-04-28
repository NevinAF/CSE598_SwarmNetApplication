import numpy


# from sklearn.preprocessing import MinMaxScaler


def preprocess_csv(data):
    # data_list = []
    steps = data["Time"].unique()
    ids = data["ID"].unique()
    steps_array = numpy.zeros([len(steps), len(ids), 5])
    for step in steps:
        # step_data = []
        step_frame = data[(data["Time"] == step)]
        agent_ids = step_frame["ID"].unique()
        # if len(agent_ids) != len(ids):
        #     continue
        for agent_id in agent_ids:
            agent_index = numpy.where(ids == agent_id)
            agent_frame = step_frame[(step_frame["ID"] == agent_id)]
            agent_x = agent_frame["X Global"].unique()[0]
            agent_y = agent_frame["Y Global"].unique()[0]
            agent_r = agent_frame["Velocity R"].unique()[0]
            agent_theta = agent_frame["Velocity Theta"].unique()[0]
            time_in_store = agent_frame["Time in Store"].unique()[0]
            coord = [agent_x, agent_y, agent_r, agent_theta, time_in_store]
            steps_array[step, agent_index] = coord
        #     step_data.append(coord)
        # data_list.append(step_data)
    # data_list = numpy.array(data_list, dtype=float)
    steps_array = numpy.swapaxes(steps_array, 0, 1)
    return steps_array


def preprocess_json(data):
    data = numpy.array(data)

    return data


def preprocess_predict_steps(data, is_test_data, steps, truth_available, test_length, predict_state_length):
    if is_test_data:
        test_length = min(test_length, data.shape[1])
        data = data[:, :test_length, ]
    if truth_available:
        truth_ends_at = data.shape[1] - steps + 1
        # Ground truth starts at 7 time-steps # TODO should be variable based on num layers and kernel size
        # Shape = [agent, condensed time-step, prediction step, state-vector]
        data_x = data[:, 0:truth_ends_at - 1, :]
        # Only care about specified values for prediction
        data_y = numpy.zeros([data_x.shape[0], data_x.shape[1], steps, predict_state_length])
        # Don't want to predict on time-steps where truth no longer available
        for i in range(0, steps):
            data_y[:, 6:, i, :] = data[:, 7 + i:truth_ends_at + i, :predict_state_length]
        data_x = numpy.swapaxes(data_x, 0, 1)
        # Shape [step, agent, predicted step, state] e.g. [0][1][0][0] is the 1-step x prediction for agent 1 at \
        #   time-step 0 (6)
        data_y = numpy.swapaxes(data_y, 0, 1)
    else:
        truth_ends_at = None
    state_length = data.shape[2]
    return data_x, data_y, state_length
