import matplotlib.pyplot as plt
import numpy


def plot(truths, predictions):
    truths = numpy.swapaxes(truths, 0, 1)
    predictions = numpy.swapaxes(predictions, 0, 1)
    for i in range(len(truths)):
        for j in range(len(truths[i])):
            agent_true = truths[i][j]
            vel_true = numpy.array([agent_true[0][2], agent_true[0][3]])
            vel_true = vel_true / numpy.linalg.norm(vel_true)
            agent_pred = predictions[i][j]
            vel_pred = numpy.array([agent_pred[0][2], agent_pred[0][3]])
            vel_pred = vel_pred / numpy.linalg.norm(vel_pred)
            if not numpy.all(agent_true[0] == 0):
                plt.arrow(agent_true[0][0], agent_true[0][1], vel_true[0], vel_true[1], width=0.05, **{'color': 'c'})
                plt.arrow(agent_pred[0][0], agent_pred[0][1], vel_pred[0], vel_pred[1], width=0.05, **{'color': 'r'})

    plt.savefig('predictions.jpg', dpi=300)
