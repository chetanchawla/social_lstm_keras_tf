#importing the libraries
import os
from argparse import Namespace, ArgumentParser
from shutil import copyfile

import matplotlib.pyplot as plt

#importing other python files as depenencies 
from data_utils import obs_pred_split
from general_utils import dump_json_file
from general_utils import now_to_str
from load_model_config import ModelConfig
from load_model_config import load_model_config
from my_social_model import MySocialModel
from provide_train_test import provide_train_test

#This is the first function called from main function
#The function parses the dataset configuration we want to use and the output directory we want to use for saving the results
def load_train_args() -> Namespace:
    parser = ArgumentParser() #an argument parser is used here
    parser.add_argument("--config", type=str, required=True)#configs file name is taken here
    parser.add_argument("--out_root", type=str, default="data/results")#output results directory is taken here
    return parser.parse_args() #The parsed arguments are returned in the program from the function here.


def _make_weights_file_name(n_epochs: int) -> str:
    return "social_train_model_e{0:04d}.h5".format(n_epochs)

#The function is used to train the model and is called right after loading the config, the ModelConfig obect storing 
#configuration attributes and out_root string address for saving output
def train_social_model(out_dir: str, config: ModelConfig) -> None:
    # load data
    train_data, test_data = provide_train_test(config)

    # prepare train data
    obs_len_train, pred_len_train = obs_pred_split(config.obs_len,
                                                   config.pred_len,
                                                   *train_data)
    x_obs_len_train, _, grid_obs_len_train, zeros_obs_len_train = obs_len_train
    _, y_pred_len_train, _, _ = pred_len_train

    # prepare test data
    obs_len_test, pred_len_test = obs_pred_split(config.obs_len,
                                                 config.pred_len,
                                                 *test_data)
    x_obs_len_test, _, grid_obs_len_test, zeros_obs_len_test = obs_len_test
    _, y_pred_len_test, _, _ = pred_len_test

    os.makedirs(out_dir, exist_ok=True)

    # training
    my_model = MySocialModel(config)
    history = my_model.train_model.fit(
        [x_obs_len_train, grid_obs_len_train, zeros_obs_len_train],
        y_pred_len_train,
        batch_size=config.batch_size,
        epochs=config.n_epochs,
        verbose=1,
        validation_data=(
            [x_obs_len_test, grid_obs_len_test, zeros_obs_len_test],
            y_pred_len_test
        )
    )

    # save loss plot
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("social model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.savefig(os.path.join(out_dir, "test={}_loss.png".format(
        config.test_dataset_kind)))

    history_file = os.path.join(out_dir, "history.json")
    dump_json_file(history.history, history_file)

    # save the trained model weights
    weights_file = os.path.join(out_dir,
                                _make_weights_file_name(config.n_epochs))
    my_model.train_model.save_weights(weights_file)

#this is the main fuction from where the excecution of the whole code starts
def main():
    args = load_train_args() #This command loads the directory in which the configs is present and loads the config file we want to use along with the directory of output files we wish to have
    config = load_model_config(args.config)#This loads the configuration chosen from the json file as an object of ModelConfig type which has all the attributes attached to it
    config.data_root = os.path.abspath(config.data_root)#This is the data root folder from where we picjed the configs from
    now_str = now_to_str()#this stores the current (now) time and date stamp as a string

    out_dir = os.path.join(args.out_root, "{}".format(now_str),
                           "test={}".format(config.test_dataset_kind))#This specifies the complete output results directory which stores
    #the current date time stamp for easy access late on at the output folder specified at the out_root

    train_social_model(out_dir, config)#calls the function train_social_model to do the training with arguments out_dir being the complete
    #output address and config being the ModelConfig object storing config attributes of the chosen configuration
    copyfile(args.config,
             os.path.join(out_dir, os.path.basename(args.config)))


if __name__ == '__main__':
    main()
