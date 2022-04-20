import torch

from swarm_gnn import retrieve_model


def start(model_path):
    model = retrieve_model(model_path)

    return model


@torch.no_grad()
def control(last_steps, model, predict_steps):
    control_vel = model.forward(last_steps, predict_steps)

    return control_vel
