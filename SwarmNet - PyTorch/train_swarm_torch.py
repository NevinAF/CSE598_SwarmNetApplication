import json
import multiprocessing
import os
import random
import time

import numpy
import torch
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler

from swarm_gnn import retrieve_model, retrieve_loss, utils, ExperimentConfig
from swarm_gnn.dataset import SimulationDataset, retrieve_test_set, retrieve_train_sets
from swarm_gnn.model import SwarmNet
from swarm_gnn.preprocessing import preprocess_predict_steps

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(epoch, model, batches, optimizer, loss_fcn, prediction_steps):
    model.train()

    loss_list = []
    # predictions = []
    # truths = []
    # batches = []
    # TODO write custom dataloader
    # Since data relies on other data in the same set, use multiple loaders with their own batches
    for batch in batches:
        X = batch[0]
        y = batch[1]
        if X.shape[0] < 7:
            continue
        # original_y = y.tolist()
        optimizer.zero_grad()
        X = X.to(device)
        y = y[6:]
        y = y.to(device)
        # original_x = X.tolist()
        # new_y = y.tolist()

        # Forward pass
        y_pred = model(X.float(), prediction_steps)[:, :, :, :model.predict_state_length]
        # test = y_pred.tolist()
        # test2 = y.float().tolist()
        loss = loss_fcn(y_pred.float(), y.float())
        # truth_list = y_pred.tolist()
        # if batch % 10 == 0:
        #     print(loss)

        # Backward pass
        loss.backward()
        optimizer.step()
        # y_pred_detach = y_pred.detach()
        # y_detach = y.detach()
        # predictions.append(y_pred_detach.tolist())
        # truths.append(y_detach.tolist())

        # Store losses for epoch
        loss_list.append(loss.cpu().item())
    # predictions = numpy.concatenate(predictions)
    # truths = numpy.concatenate(truths)
    # metrics = utils.metrics(predictions, truths)

    return loss_list


@torch.no_grad()
def validate(epoch, model, dataset, optimizer, loss_fcn, config):
    model.eval()
    loss_list = []
    predictions = []
    truths = []
    for batch, (X, y) in enumerate(dataset):
        if X.shape[0] < 7:
            continue
        X = X.to(device)
        y = y[6:]
        y = y.to(device)

        # Forward pass
        y_pred = model(X.float(), dataset.dataset.prediction_steps)
        y_pred_detach = y_pred.cpu().detach()
        loss = loss_fcn(y_pred, y)
        predictions.append(y_pred_detach.numpy())
        truths.append(y.numpy())

        # Store losses for epoch
        loss_list.append(loss.cpu().item())
    predictions = numpy.concatenate(predictions)

    return loss_list


@torch.no_grad()
def test(epoch, model, dataset, loss_fcn, config):
    model.eval()
    losses = []
    predictions = []
    truths = []
    for batch, (X, y) in enumerate(dataset):
        # Minimum 7 steps + 1 additional step for each additional prediction step
        if X.shape[0] < 7:
            continue
        X = X.to(device)
        y = y[6:]
        # y = torch.tensor(y.tolist()[6:])
        y = y.to(device)
        y_pred = model(X.float(), dataset.dataset.prediction_steps)[:, :, :, :model.predict_state_length]
        if config.truth_available:
            loss = loss_fcn(y_pred, y)
            losses.append(loss.cpu().item())
        predictions.append(y_pred.cpu().detach().numpy())
        truths.append(y.cpu().detach().numpy())
    predictions = numpy.concatenate(predictions)
    truths = numpy.concatenate(truths)
    # print(epoch)
    # metrics = utils.metrics(predictions, truths)

    return losses, predictions, truths


def train_mode(config):
    train_loaders = []
    val_loaders = []
    if device == 'cuda':
        num_workers = torch.cuda.device_count()
    else:
        num_workers = 0

    # Initialize the model
    if config.load_train is False:
        # Initialize the dataset
        test_set = retrieve_test_set(config, scaler=None, predict_steps=config.prediction_steps)
        train_sets = retrieve_train_sets(config.train_paths, config, scaler=None, predict_steps=config.prediction_steps)
        model = SwarmNet(train_sets[0].state_length, config.predict_state_length)
    elif config.model_load_path is not None:
        model = retrieve_model(config.model_load_path)
        if config.curriculum is True:
            # Initialize the dataset
            test_set = retrieve_test_set(config, scaler=None, predict_steps=model.predictions_trained_to)
            train_sets = retrieve_train_sets(config.train_paths, scaler=None,
                                             predict_steps=model.predictions_trained_to)

    else:
        print("Need model path if training existing model. Either povide path or set load_train to False")
        exit(0)
    model.scaler = train_sets[0].scaler
    model = model.to(device)

    for train_set in train_sets:
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
                                  sampler=train_sampler, num_workers=num_workers)
        val_loaders.append(validation_loader)
        train_loaders.append(train_loader)

    # # Test loader
    test_loader = DataLoader(test_set, batch_size=config.batch_size, num_workers=num_workers)

    optimizer_opts = {"lr": 1e-1, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-5}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Initialize the loss
    loss_fcn = retrieve_loss(config.loss_name)
    model_path = config.model_save_path
    epochs_low_loss_diff = 0
    last_val_loss = 999999
    last_test_loss = 999999
    min_epochs = config.min_epochs_per_curric
    curriculum_epoch_num = 0
    batches = []
    for loader in train_loaders:
        batch_set = list(loader)
        # Merge all batches into on set of batches, each batch only containing samples from one dataset
        batches.extend(batch_set)
    # Shuffle batches
    random.shuffle(batches)
    model_checkpoint = model.state_dict()

    # Epoch Training loop
    for epoch in range(config.epochs):
        curriculum_epoch_num += 1
        lowest_mse = model.lowest_mse_this_horizon
        time_start = time.time()
        print(epoch)
        # Train for one epoch
        loss_train = train(epoch, model, batches, optimizer, loss_fcn, loader.dataset.prediction_steps)
        loss_train = numpy.mean(loss_train)
        print(loss_train)
        # print("diff:")
        # print(loss_diff)
        # loss_validate = validate(epoch, model, validation_loader, optimizer, loss_fcn, config)
        # loss_validate = numpy.mean(loss_validate)
        # print(loss_validate)
        #
        # Test for
        loss_test, predictions, truths = test(epoch, model, test_loader, loss_fcn, config)
        loss_test = numpy.mean(loss_test)
        print(predictions[0][0])
        print(truths[0][0])
        print(loss_test)
        # Validation convergence
        loss_diff = abs(loss_test - last_test_loss)
        time_end = time.time()
        time_taken = time_end - time_start
        print(time_taken)
        # TODO save different model per horizon
        if loss_test < lowest_mse:
            print("New lowest MSE, saving model")
            lowest_mse = loss_test
            model.lowest_mse_this_horizon = lowest_mse
            torch.save(model, model_path)
            model_checkpoint = model.state_dict()
            # plotVis(test_set.data, predictions, truths)
        if loss_diff <= 0.001:
            epochs_low_loss_diff += 1
        else:
            epochs_low_loss_diff = 0
        # If train converging and test not improving
        # TODO better curriculum update criteria. Arbitrary epochs and convergence? Increase num required epochs?
        if epochs_low_loss_diff > 5 and loss_test > last_test_loss:
            epochs_low_loss_diff = 0
            if config.curriculum is True and curriculum_epoch_num > min_epochs:
                print("--------------------------UPDATING CURRICULUM--------------------------")
                curriculum_epoch_num = 0
                # if epoch > 0 and epoch % 10 == 0 and config.prediction_steps < 10:
                if model.predictions_trained_to < config.max_curric_steps:
                    model.load_state_dict(model_checkpoint)
                    train_sets, test_set, train_loaders, test_loader, validation_loaders = \
                        update_curriculum(train_sets, test_set, config, num_workers)
                    batches = []
                    for loader in train_loaders:
                        batch_set = list(loader)
                        # Merge all batches into on set of batches, each batch only containing samples from one dataset
                        batches.extend(batch_set)
                    # Shuffle batches
                    random.shuffle(batches)
                    # TODO Start from model with lowest test MSE
                    # model = retrieve_model(model_path)
                    model.predictions_trained_to += 1
                    print("Training step: " + str(model.predictions_trained_to))
                    model.lowest_mse_this_horizon = 999999
                else:
                    print("MAX CURRICULUM REACHED. CONVERGENCE LIKELY. CONSIDER STOPPING")

        last_val_loss = loss_train
        last_test_loss = loss_test

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
    test_dataset = SimulationDataset(config.test_path, True, model.scaler, config, config.prediction_steps)
    test_dataset.prediction_steps = config.prediction_steps
    test_loader = DataLoader(test_dataset, batch_size=99999999)
    loss_fcn = retrieve_loss(config.loss_name)
    loss_test, predictions, truths = test(0, model, test_loader, loss_fcn, config)
    print(predictions[0][0])
    print(truths[0][0])
    predictions = numpy.swapaxes(predictions, 0, 1)
    truths = numpy.swapaxes(truths, 0, 1)
    print(numpy.mean(loss_test))
    with open('predictions.json', 'w') as outfile:
        json.dump(predictions.tolist(), outfile)


def update_curriculum(train_sets, test_set, config, num_workers):
    train_loaders = []
    validation_loaders = []
    test_set.prediction_steps += 1
    for train_set in train_sets:
        train_set.prediction_steps += 1
        train_set.data_x, train_set.data_y, train_set.state_length = \
            preprocess_predict_steps(train_set.original_data, False,
                                     train_set.prediction_steps,
                                     config.truth_available, config.test_seg_length, config.predict_state_length)
        train_set.X = torch.tensor(train_set.data_x)
        train_set.y = torch.tensor(train_set.data_y)
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
                                  sampler=train_sampler, num_workers=num_workers, persistent_workers=False)
        train_loaders.append(train_loader)
        validation_loaders.append(validation_loader)
    test_set.data_x, test_set.data_y, test_set.state_length = \
        preprocess_predict_steps(test_set.original_data, True,
                                 test_set.prediction_steps,
                                 config.truth_available, config.test_seg_length, config.predict_state_length)
    test_set.X = torch.tensor(test_set.data_x)
    test_set.y = torch.tensor(test_set.data_y)

    # # Test loader
    test_loader = DataLoader(test_set, batch_size=config.batch_size, num_workers=num_workers, persistent_workers=False)

    return train_sets, test_set, train_loaders, test_loader, validation_loaders


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
