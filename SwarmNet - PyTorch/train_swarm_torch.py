import json
import multiprocessing
import os

import numpy
import torch
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler

from swarm_gnn import retrieve_model, retrieve_loss, retrieve_dataset, utils, ExperimentConfig
from swarm_gnn.dataset import SimulationDataset
from swarm_gnn.model import SwarmNet
from swarm_gnn.preprocessing import preprocess_predict_steps

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# TODO testing
device = 'cpu' if torch.cuda.is_available() else 'cpu'


def train(epoch, model, dataset, optimizer, loss_fcn, config):
    model.train()

    loss_list = []
    predictions = []
    truths = []

    for batch, (X, y) in enumerate(dataset):
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)

        # Forward pass
        y_pred = model(X.float(), config.prediction_steps)
        test = y_pred.tolist()
        loss = loss_fcn(y_pred, y.float())
        # truth_list = y_pred.tolist()
        # if batch % 10 == 0:
        #     print(loss)

        # Backward pass
        loss.backward()
        optimizer.step()
        y_pred_detach = y_pred.detach()
        y_detach = y.detach()
        predictions.append(y_pred_detach.tolist())
        truths.append(y_detach.tolist())

        # Store losses for epoch
        loss_list.append(loss.cpu().item())
    predictions = numpy.concatenate(predictions)
    truths = numpy.concatenate(truths)
    # metrics = utils.metrics(predictions, truths)

    return loss_list


def validate(epoch, model, dataset, optimizer, loss_fcn, config):
    model.eval()
    loss_list = []
    predictions = []
    truths = []
    for batch, (X, y) in enumerate(dataset):
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)

        # Forward pass
        y_pred = model(X.float())
        y_pred_detach = y_pred.cpu().detach()
        loss = loss_fcn(y_pred, y)
        predictions.append(y_pred_detach.numpy())
        truths.append(y.numpy())

        # Store losses for epoch
        loss_list.append(loss.cpu().item())
    predictions = numpy.concatenate(predictions)

    return loss_list


def test(epoch, model, dataset, loss_fcn, config):
    model.eval()
    losses = []
    accuracy = []
    f1 = []
    precision = []
    specificity = []
    summary = {}
    predictions = []
    truths = []
    for batch, (X, y) in enumerate(dataset):
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X.float(), config.prediction_steps).cpu().detach()
        print(y[0][0])
        print(y_pred[0][0])
        if config.truth_available:
            loss = loss_fcn(y_pred, y)
            losses.append(loss.cpu().item())
        predictions.append(y_pred.numpy())
        truths.append(y.numpy())
    predictions = numpy.concatenate(predictions)
    truths = numpy.concatenate(truths)
    # print(epoch)
    # metrics = utils.metrics(predictions, truths)

    return losses, predictions


def train_mode(config):
    # Initialize the dataset
    train_set, test_set = retrieve_dataset(config, scaler=None)

    # Validation loader
    validation_split = 0
    dataset_length = len(train_set)
    indices = list(range(dataset_length))
    validation_length = int(numpy.floor(validation_split * dataset_length))
    validation_index = numpy.random.choice(indices, size=validation_length, replace=False)
    validation_sampler = SequentialSampler(validation_index)
    validation_loader = DataLoader(train_set, batch_size=config.batch_size, sampler=validation_sampler)

    # Train loader
    train_index = list(set(indices) - set(validation_index))
    train_sampler = SequentialSampler(train_index)
    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              sampler=train_sampler)  # , num_workers=torch.cuda.device_count())

    # # Test loader
    test_loader = DataLoader(test_set, batch_size=config.batch_size)

    # Initialize the model
    if config.load_train is False:
        # TODO filled in placeholders for testing
        time_steps = 2000
        model = SwarmNet(train_set.state_length)
    elif config.model_load_path is not None:
        model = retrieve_model(config.model_load_path)
    else:
        print("Need model path if training existing model. Either provide path or set load_train to False")
        exit(0)
    model.scaler = train_set.scaler
    model = model.to(device)
    optimizer_opts = {"lr": 1e-1, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-5}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Initialize the loss
    loss_fcn = retrieve_loss(config.loss_name)
    lowest_mse = model.lowest_mse
    model_path = config.model_save_path
    epochs_low_loss = 0
    last_val_loss = 999999
    last_test_loss = 0
    save_loc = os.path.join(os.getcwd(), "model.pkl")

    # Epoch Training loop
    for epoch in range(config.epochs):
        print(epoch)
        # Train for one epoch
        loss_train = train(epoch, model, train_loader, optimizer, loss_fcn, config)
        loss_train = numpy.mean(loss_train)
        loss_diff = abs(loss_train - last_val_loss)
        last_val_loss = loss_train
        print("diff:")
        print(loss_diff)
        # loss_validate = validate(epoch, model, validation_loader, optimizer, loss_fcn, config)
        # loss_validate = numpy.mean(loss_validate)
        # print(loss_validate)
        #
        # Test for
        loss_test, predictions = test(epoch, model, test_loader, loss_fcn, config)
        loss_test = numpy.mean(loss_test)
        if loss_test < lowest_mse and config.prediction_steps >= 10:
            print("New lowest MSE, saving model")
            lowest_mse = loss_test
            model.lowest_mse = lowest_mse
            torch.save(model, save_loc)
            # plotVis(test_set.data, predictions, truths)
        if loss_diff <= 0.01:
            epochs_low_loss += 1
        else:
            epochs_low_loss = 0
        if epochs_low_loss > 2:
            epochs_low_loss = 0
            last_val_loss = 999999
            print("--------------------------UPDATING CURRICULUM--------------------------")
            if config.curriculum is True:
                # if epoch > 0 and epoch % 10 == 0 and config.prediction_steps < 10:
                if config.prediction_steps < 10:
                    config.prediction_steps += 1
                    train_loader.dataset.data_x, train_loader.dataset.data_y, train_loader.dataset.state_length = \
                        preprocess_predict_steps(train_loader.dataset.original_data, False, config.prediction_steps,
                                                 config.truth_available)
                    train_loader.dataset.X = torch.tensor(train_loader.dataset.data_x)
                    train_loader.dataset.y = torch.tensor(train_loader.dataset.data_y)
                    test_loader.dataset.data_x, test_loader.dataset.data_y, test_loader.dataset.state_length = \
                        preprocess_predict_steps(test_loader.dataset.original_data, True, config.prediction_steps,
                                                 config.truth_available)
                    test_loader.dataset.X = torch.tensor(test_loader.dataset.data_x)
                    test_loader.dataset.y = torch.tensor(test_loader.dataset.data_y)

        print(loss_train)


        # f1 = metrics_test[1]
        # if f1 > highest_f1:
        #     print("New highest f1 score, saving model")
        #     highest_f1 = f1
        #     model.highest_test_score = highest_f1
        #     torch.save(model, model_path)
        #     # plotVis(test_set.data, predictions, truths)

        # val_loss_diff = abs(loss_validate - last_val_loss)
        # last_val_loss = loss_validate

        # test_loss_diff = abs(loss_test - last_test_loss)
        # last_test_loss = loss_test

    # print(loss)
    # print(summary)


def test_mode(config):
    if config.model_load_path is not None:
        model = retrieve_model(config.model_load_path)
    else:
        print("Need existing model path to test model")
        exit(0)
    test_dataset = SimulationDataset(config.test_path, True, model.scaler, config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    loss_fcn = retrieve_loss(config.loss_name)
    loss_test, predictions = test(0, model, test_loader, loss_fcn, config)
    print(numpy.mean(loss_test))
    predictions_json = json.dumps(predictions.tolist())
    with open('predictions.json', 'w') as outfile:
        json.dump(predictions_json, outfile)


def main():
    # Initialize the config class
    config_file = os.path.join(os.getcwd(), 'configs', 'config.yaml')
    config = ExperimentConfig(config_file)

    if config.mode == "train":
        train_mode(config)

    if config.mode == "test":
        test_mode(config)


if __name__ == '__main__':
    main()
