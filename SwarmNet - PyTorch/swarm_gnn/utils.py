import os

import matplotlib.pyplot as plt
import numpy
from matplotlib import image as mpimg


def plot(truths, predictions):
    truths = numpy.swapaxes(truths, 0, 1)
    predictions = numpy.swapaxes(predictions, 0, 1)
    img_path = os.path.join(os.getcwd(), 'environment.bmp')
    img = mpimg.imread(img_path)  # read the image
    # TODO specify ground truth range
    for i in range(0, 294):
        # TODO specify agent range
        for j in range(1, 3):
            agent_true = truths[i][j]
            vel_true = numpy.array([agent_true[0][2], agent_true[0][3]])
            vel_true = vel_true / numpy.linalg.norm(vel_true)
            if not numpy.all(agent_true[0] == 0):
                plt.arrow(agent_true[0][0], agent_true[0][1], vel_true[0], vel_true[1], width=0.05, **{'color': 'c'})
                # TODO specify range to output multistep prediction or single step
                if i == 10:
                    agent_pred = predictions[i][j]
                    for prediction_step in range(agent_pred.shape[0]):
                        if not numpy.all(agent_true[prediction_step] == 0):
                            vel_pred = numpy.array([agent_pred[prediction_step][2], agent_pred[prediction_step][3]])
                            vel_pred = vel_pred / numpy.linalg.norm(vel_pred)
                            plt.arrow(agent_pred[prediction_step][0], agent_pred[prediction_step][1], vel_pred[0], vel_pred[1], width=0.05, **{'color': 'r'})
    plt.imshow(img)

    plt.savefig('predictions.jpg', dpi=1000)
