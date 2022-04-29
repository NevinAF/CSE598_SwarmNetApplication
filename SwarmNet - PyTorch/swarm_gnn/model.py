import numpy
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def retrieve_model(path):
    model = torch.load(path, map_location=device)
    model.eval()
    return model


class SwarmNet(nn.Module):
    def __init__(self, agent_state_vector_length, predict_state_length):
        super(SwarmNet, self).__init__()
        self.conv_layer1 = nn.Conv1d(agent_state_vector_length, 32, kernel_size=3, groups=1)
        self.conv_layer2 = nn.Conv1d(32, 64, kernel_size=3, groups=1)
        self.conv_layer3 = nn.Conv1d(64, 32, kernel_size=3, groups=1)
        self.edge_state_mlp = nn.Linear(32 * 2, 32)
        self.edge_agg_mlp = nn.Linear(32, 32)
        self.node_updater_mlp = nn.Linear(32 * 2, 32)
        self.node_decoder_mlp = nn.Linear(32, predict_state_length)
        # self.node_decoder_mlp.requires_grad_(False)
        self.relu = nn.ReLU()
        self.scaler = None
        self.predictions_trained_to = 1
        self.predict_state_length = predict_state_length
        self.lowest_mse_this_horizon = 999999999999

    def conv_1d(self, x, predict_step):
        if predict_step == 0:
            agent_transpose = torch.transpose(x, 1, 2)
            condense_1 = self.conv_layer1(agent_transpose)
            condense_1 = self.relu(condense_1)
            condense_2 = self.conv_layer2(condense_1)
            condense_2 = self.relu(condense_2)
            condense_3 = self.conv_layer3(condense_2)
            condense_3 = self.relu(condense_3)
            # Reshape into [agent, condensed time-step, condensed state vector]
            condense_3 = torch.transpose(condense_3, 1, 2)
            return condense_3
        else:
            # Convolve over each window from previous prediction steps
            agent_transpose = torch.transpose(x, 2, 3)
            num_agents = x.shape[0]
            num_steps = x.shape[1]
            agent_transpose = torch.flatten(agent_transpose, start_dim=0, end_dim=1).float()
            condense_1 = self.conv_layer1(agent_transpose)
            condense_1 = self.relu(condense_1)
            condense_2 = self.conv_layer2(condense_1)
            condense_2 = self.relu(condense_2)
            condense_3 = self.conv_layer3(condense_2)
            condense_3 = self.relu(condense_3)
            # Reshape into [agent, condensed time-step, condensed state vector]
            condense_3 = torch.reshape(condense_3, [num_agents, num_steps, condense_3.shape[1]])
            return condense_3

    def graph_conv(self, condensed_steps):
        num_nodes = condensed_steps.shape[1]
        outgoing_messages = torch.repeat_interleave(torch.unsqueeze(
            condensed_steps, 2), num_nodes, dim=2)
        incoming_messages = torch.repeat_interleave(torch.unsqueeze(
            condensed_steps, 1), num_nodes, dim=1)
        # Shape [Condensed steps, agents, incoming agents, concatenated state vectors]
        node_msgs = torch.concat([incoming_messages, outgoing_messages], dim=-1)
        edges = self.edge_state_mlp(node_msgs)
        edges = self.relu(edges)
        agg_edges = torch.sum(edges, dim=2)
        agg_edges = self.edge_agg_mlp(agg_edges)
        agg_edges = self.relu(agg_edges)
        updated_nodes = torch.concat([condensed_steps, agg_edges], dim=-1)
        updated_nodes = self.node_updater_mlp(updated_nodes)
        updated_nodes = self.relu(updated_nodes)
        return updated_nodes

    def decode(self, predict, x, predict_step):
        original = x
        original = torch.swapaxes(original, 0, 1)
        decoded_states = self.node_decoder_mlp(predict)
        if predict_step == 0:
            original_states = original[6:, :, :]
        else:
            original_states = original[:, :, 6, :]
        # Add predicted change to original state
        decoded_steps = torch.add(original_states[:, :, :self.predict_state_length], decoded_states)
        # Concatenate additional information not being predicted for next steps of predictions
        decoded_steps = torch.concat([decoded_steps, original_states[:, :, self.predict_state_length:]], axis=2)
        # # TODO temporarily not predicting obstacles (nodes 50+ for this dataset) Hard coding this is not good
        # decoded_steps[:, 50:, :] = original_states[:, 50:, :]
        # decoded_steps_arr = numpy.array(decoded_steps.tolist())
        # original_steps_arr = numpy.array(original.tolist())
        return decoded_steps

    def forward(self, x, predict_steps):
        x = numpy.swapaxes(x, 0, 1)
        condensed_steps = x
        all_predict = []
        for i in range(0, predict_steps):
            condensed_agents = self.conv_1d(condensed_steps, i)
            # Reshaped to [time-steps, agents, condensed_state_length]
            time_steps = torch.transpose(condensed_agents, 0, 1)
            predict = self.graph_conv(time_steps)
            # Shape [condensed time-steps, agents, predicted state vector]
            predict = self.decode(predict, condensed_steps, i)
            # Shape [agents, condensed time-steps, predicted state vector]
            predict = torch.swapaxes(predict, 0, 1)
            extend_pred_list = numpy.zeros([predict.size(0), predict.size(1), 7, predict.size(2)])
            test_list = numpy.array(condensed_steps.tolist())
            extend_pred = torch.tensor(extend_pred_list)
            extend_pred = extend_pred.to(device)
            if predict_steps > 1:
                # On first prediction, reshape to keep track of original window that was condensed
                if i == 0:
                    for j in range(0, predict.size(0)):
                        for z in range(0, predict.size(1)):
                            extend_pred_list[j, z, :, :] = test_list[j, z:z + 7, :]
                            extend_pred[j, z, :, :] = condensed_steps[j, z:z + 7, :]
                else:
                    # extend_pred_list = test_list
                    extend_pred = condensed_steps

                # pre_roll = extend_pred.tolist()
                # Roll window to left
                extend_pred = torch.roll(extend_pred, -1, 2)
                # Replace last step in window (first step before roll) with previous prediction
                extend_pred[:, :, 6, :] = predict[:, :, :]
                # test2 = extend_pred.tolist()
                condensed_steps = extend_pred
                # test3 = predict.tolist()
            all_predict.append(predict)

        # Shape = [agent, condensed time-step, prediction step, state-vector]
        predict = torch.stack(all_predict, 2)
        # Shape = [condensed time-step, agent, prediction step, state-vector]
        predict = torch.swapaxes(predict, 0, 1)
        # test = predict.tolist()

        return predict
