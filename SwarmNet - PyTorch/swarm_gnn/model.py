import numpy
import torch
from torch import nn


def retrieve_model(path):
    model = torch.load(path)
    model.eval()
    return model


class SwarmNet(nn.Module):
    def __init__(self, agent_state_vector_length):
        super(SwarmNet, self).__init__()
        # state vector, num output filters, kernel size, groups= state vector
        self.conv_layer1 = nn.Conv1d(agent_state_vector_length, 16, kernel_size=3, groups=1)
        self.conv_layer2 = nn.Conv1d(16, 32, kernel_size=3, groups=1)
        self.conv_layer3 = nn.Conv1d(32, 32, kernel_size=3, groups=1)
        self.edge_state_mlp = nn.Linear(32 * 2, 32)
        self.edge_agg_mlp = nn.Linear(32, 32)
        self.node_updater_mlp = nn.Linear(32 * 2, 32)
        self.node_decoder_mlp = nn.Linear(32, agent_state_vector_length)
        self.relu = nn.ReLU()
        self.scaler = None
        self.lowest_mse = 999999999999

    def conv_1d(self, x, predict_step):
        # Condense by agent individually. Now only requires set-length state vector as input while agent numbers can
        #   vary (between runs. Agent count must be static once started).
        #   May also allow for eventually handling dynamically spawning agents
        condensed_agents_list = []
        # On first prediction, convolve over all steps for agent
        if predict_step == 0:
            for agent in x:
                agent_transpose = torch.transpose(agent, 0, 1)
                condense_1 = self.conv_layer1(agent_transpose)
                condense_1 = self.relu(condense_1)
                condense_2 = self.conv_layer2(condense_1)
                condense_2 = self.relu(condense_2)
                condense_3 = self.conv_layer3(condense_2)
                # TODO often don't see activation on last layer. Unsure if holds when moving from one type of layer to
                #   another. Possibly remove this in future
                condense_3 = self.relu(condense_3)
                # Reshape into [Condensed time-steps, condensed state vector]
                condense_3 = torch.transpose(condense_3, 0, 1)
                condensed_agents_list.append(condense_3)
        else:
            # Convolve over each window from previous prediction steps
            for step in x:
                local_agent = []
                for agent in step:
                    agent_transpose = torch.transpose(agent, 0, 1)
                    condense_1 = self.conv_layer1(agent_transpose.float())
                    condense_1 = self.relu(condense_1)
                    condense_2 = self.conv_layer2(condense_1)
                    condense_2 = self.relu(condense_2)
                    condense_3 = self.conv_layer3(condense_2)
                    # TODO often don't see activation on last layer. Unsure if holds when moving from one type of
                    #  layer to another. Possibly remove this in future
                    condense_3 = self.relu(condense_3)
                    # Reshape into [Condensed time-steps, condensed state vector]
                    condense_3 = torch.transpose(condense_3, 0, 1)
                    local_agent.append(torch.squeeze(condense_3))
                local_agent = torch.stack(local_agent)
                test = local_agent.tolist()
                condensed_agents_list.append(local_agent)

        # Condensed to shape [Agents, condensed time-steps, condensed_state_length]
        condensed_agents = torch.stack(condensed_agents_list)

        return condensed_agents

    # TODO given that the predictions for each condensed steps are independent and that this is quite slow for a lot
    #   of steps, multiprocessing might be a good thing to implement here. Divide steps by number of available cores,
    #   run calculations, wait for all to return, condense in proper order. Issue: will this break autograd graph?
    def graph_conv(self, condensed_steps):
        predictions = []
        # For every condensed step
        steps = 0
        for step in condensed_steps:
            updated_nodes = []
            steps += 1
            # print(steps)
            # For every node
            for i in range(0, len(step)):
                node_i = step[i]
                agg_edge = torch.zeros(len(node_i))
                # For every other node
                for j in range(0, len(step)):
                    # Ignore node if same node
                    if i == j:
                        continue
                    node_j = step[j]
                    # Message from node j to node i
                    edge_j_i = torch.concat((node_j, node_i))
                    # Edge state from node j to node i
                    edge_j_i = self.edge_state_mlp(edge_j_i)
                    edge_j_i = self.relu(edge_j_i)
                    # Aggregate all incoming edges
                    agg_edge = torch.add(agg_edge, edge_j_i)
                # Aggregated edge state from all node j to node i
                agg_edge = self.edge_agg_mlp(agg_edge)
                agg_edge = self.relu(agg_edge)
                # Update node i and maintain in separate list to not affect messages between other nodes in this conv
                i_updated = torch.concat((node_i, agg_edge))
                i_updated = self.node_updater_mlp(i_updated)
                i_updated = self.relu(i_updated)
                updated_nodes.append(i_updated)
            updated_nodes = torch.stack(updated_nodes)
            predictions.append(updated_nodes)
        # Shape [time-step, agent, predicted condensed state]
        predictions = torch.stack(predictions)
        return predictions

    def decode(self, predict, x):
        original = x
        decoded_steps = []
        # For every predicted time-step
        for t in range(0, len(predict)):
            step = predict[t]
            decoded_states = []
            # For every agent
            for i in range(0, len(step)):
                node_i = predict[t][i]
                # Decode back into original dimensionality (agent state vector)
                decoded_i = self.node_decoder_mlp(node_i)
                original_state = original[i][6 + t]
                # Decoded state is prediction of change. Add to state prediction stemmed from
                decoded_states.append(torch.add(original_state, decoded_i))
            decoded_states = torch.stack(decoded_states)
            decoded_steps.append(decoded_states)
        decoded_steps = torch.stack(decoded_steps)
        return decoded_steps

    def forward(self, x, predict_steps):
        condensed_steps = x
        all_predict = []
        for i in range(0, predict_steps):
            condensed_agents = self.conv_1d(condensed_steps, i)
            # Reshaped to [time-steps, agents, condensed_state_length]
            time_steps = torch.transpose(condensed_agents, 0, 1)
            predict = self.graph_conv(time_steps)
            # Shape [condensed time-steps, agents, predicted state vector]
            predict = self.decode(predict, x)
            # Shape [agents, condensed time-steps, predicted state vector]
            predict = torch.swapaxes(predict, 0, 1)
            extend_pred_list = numpy.zeros([predict.size(0), predict.size(1), 7, predict.size(2)])
            test_list = numpy.array(condensed_steps.tolist())
            extend_pred = torch.tensor(extend_pred_list)
            if predict_steps > 1:
                # On first prediction, reshape to keep track of original window that was condensed
                if i == 0:
                    for j in range(0, predict.size(0)):
                        for z in range(0, predict.size(1)):
                            extend_pred_list[j, z, :, :] = test_list[j, z:z + 7, :]
                            extend_pred[j, z, :, :] = condensed_steps[j, z:z + 7, :]
                else:
                    extend_pred_list = test_list
                    extend_pred = condensed_steps

                # Roll window to left
                extend_pred = torch.roll(extend_pred, -1, 2)
                # Replace last step in window (first step before roll) with previous prediction
                extend_pred[:, :, 6, :] = predict[:, :, :]
                test2 = extend_pred.tolist()
                condensed_steps = extend_pred
                test3 = predict.tolist()
            all_predict.append(predict)

        # Shape = [agent, condensed time-step, prediction step, state-vector]
        predict = torch.stack(all_predict, 2)
        test = predict.tolist()

        return predict
