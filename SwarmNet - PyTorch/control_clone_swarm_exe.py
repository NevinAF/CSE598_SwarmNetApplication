import json
import numpy
import torch
import sys

def start(model_path):
	model = torch.load(model_path, map_location='cpu')

	return model


@torch.no_grad()
def control(last_steps, model, predict_steps):
	device = 'cpu'
	last_steps = numpy.asarray(last_steps, dtype=float)
	last_steps = numpy.nan_to_num(last_steps)
	last_steps = numpy.swapaxes(last_steps, 0, 1)
	if last_steps.shape[1] >= 7:
		last_steps = last_steps[:, :, :]
		last_steps = torch.tensor(last_steps)
		last_steps = last_steps.to(device)
		control_vel = model.forward(last_steps.float(), predict_steps)
		control_vel = control_vel.cpu().detach().numpy()
		control_vel = numpy.swapaxes(control_vel, 0, 1)
	else:
		control_vel = numpy.zeros([last_steps.shape[0], 1, last_steps.shape[2]])
		control_vel[:, :, :] = last_steps[:, 0:1, :]

	return control_vel

with open(sys.argv[1]) as f:
	data = json.load(f)
	predictions = control(data, start(sys.argv[2]), int(sys.argv[3]))

	# predictions = numpy.swapaxes(predictions, 0, 1)
	with open(sys.argv[1], 'w') as outfile:
		json.dump(predictions.tolist(), outfile, indent=1)