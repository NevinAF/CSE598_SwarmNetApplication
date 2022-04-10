import yaml


class ExperimentConfig:

    def __init__(self, config_file):
        config = yaml.safe_load(open(config_file, 'r'))
        self.prediction_steps = config["general"]["prediction_steps"]

        # Location of train dataset
        self.train_path = config["train"]["data_path"]
        self.loss_name = config["train"]["loss"]
        self.epochs = config["train"]["n_epochs"]
        self.batch_size = config["train"]["batch_size"]
        self.sample_size = config["train"]["sample_size"]
        # Add noise to training data
        self.add_noise_train = config["train"]["add_noise"]

        # Location of test data
        self.test_path = config["test"]["data_path"]
        self.test_env_width = config["test"]["env_width"]
        self.test_env_height = config["test"]["env_height"]
        # Whether ground truth is available
        self.truth_available = config["test"]["truth_available"]
        # Add noise to testing data
        self.add_noise_test = config["test"]["add_noise"]

        self.model_load_path = config["model"]["load_path"]
        # Train or test existing model
        self.mode = config["model"]["mode"].lower()
        self.load_train = config["model"]["load_train"]
        self.model_save_path = config["model"]["save_path"]

