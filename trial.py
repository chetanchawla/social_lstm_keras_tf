from argparse import Namespace, ArgumentParser
import os
import json
from datetime import datetime
#This is the first function called from main function
#The function parses the dataset configuration we want to use and the output directory we want to use for saving the results
def load_train_args() -> Namespace:
    parser = ArgumentParser() #an argument parser is used here
    parser.add_argument("--config", type=str, required=True)#configs file name is taken here
    parser.add_argument("--out_root", type=str, default="data/results")#output results directory is taken here
    return parser.parse_args() #The parsed arguments are returned in the program from the function here.
def now_to_str(format: str = "%Y%m%d%H%M%S") -> str:
    return datetime.now().strftime(format)

class ModelConfig:
    def __init__(self, n_epochs, batch_size, obs_len, pred_len,
                 max_n_peds, n_neighbor_pixels, grid_side, lstm_state_dim,
                 emb_dim, data_root, test_dataset_kind, **kwargs):
        # train config
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # model config
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.max_n_peds = max_n_peds
        self.n_neighbor_pixels = n_neighbor_pixels
        self.grid_side = grid_side
        self.grid_side_squared = grid_side ** 2

        # layer config
        self.lstm_state_dim = lstm_state_dim
        self.emb_dim = emb_dim

        # dataset config
        self.data_root = data_root
        self.test_dataset_kind = test_dataset_kind


def load_model_config(config_file: str) -> ModelConfig:
    with open(config_file, "r") as f:
        config = json.load(f)
    return ModelConfig(**config)
parsed=load_train_args()
print(parsed)
config = load_model_config(parsed.config)
config.data_root = os.path.abspath(config.data_root)
print("\n")
print("config",config, "\n")

print("config.data_root",config.data_root)
print("epochs",config.n_epochs)
print("batch size",config.batch_size)
print("oservation length",config.obs_len)
print("prediction sequence length",config.pred_len)
now_str = now_to_str()


import numpy as np
from enum import unique, Enum, auto
from typing import Union, List

from general_utils import DatasetKind
from general_utils import _check_dataset_kind
from general_utils import get_data_dir
from load_dataset import load_dataset_from_config, load_dataset
from load_model_config import ModelConfig


@unique
class DatasetKind(Enum):
    """Human-trajectory datasets used in the Social Model paper."""
    eth = auto()
    hotel = auto()
    zara1 = auto()
    zara2 = auto()
    ucy = auto()

_rel_data_dir_dict = {
    DatasetKind.eth: "eth/univ",
    DatasetKind.hotel: "eth/hotel",
    DatasetKind.zara1: "ucy/zara/zara01",
    DatasetKind.zara2: "ucy/zara/zara02",
    DatasetKind.ucy: "ucy/univ"
}
def _check_dataset_kind(dataset_kind: Union[DatasetKind, str]) -> DatasetKind:
    if isinstance(dataset_kind, DatasetKind):
        return dataset_kind

    if isinstance(dataset_kind, str) and hasattr(DatasetKind, dataset_kind):
        return DatasetKind[dataset_kind]

    raise ValueError("Unknown test_dataset_kind: {}".format(dataset_kind))

def get_data_dir(root: str, dataset_kind: Union[DatasetKind, str]) -> str:
    dataset_kind = _check_dataset_kind(dataset_kind)
    data_dir = os.path.join(root, _rel_data_dir_dict[dataset_kind])
    return data_dir



#train_data, test_data = provide_train_test(config)
all_dataset_kinds = set(DatasetKind)
print(all_dataset_kinds)
test_dataset_kind = _check_dataset_kind(config.test_dataset_kind)
train_dataset_kinds = all_dataset_kinds - {test_dataset_kind}
print("test_dataset_kind", test_dataset_kind)
print("train_dataset_kind", train_dataset_kinds)

print(Union[DatasetKind, str])


# load (several) train datasets
#make 4 empty arrays to append training data into- x_train for data points, y_train to the output sequences
#grid_train saves the grid boundary info (m,n,:) and zeros array
x_train, y_train, grid_train, zeros_train = [], [], [], []
for train_dataset_kind in train_dataset_kinds:
    data_dir = get_data_dir(config.data_root, train_dataset_kind)
    print("data_dir",data_dir)

    train_dataset = load_dataset("C:/Users/asd/Desktop/Trajectory Prediction/social_lstm_keras_tf-master/data/datasets/ucy/zara/zara01", train_dataset_kind,
                                     config.obs_len + config.pred_len,
                                     config.max_n_peds,
                                     config.n_neighbor_pixels, config.grid_side)

    x, y, g, z = train_dataset.get_data(config.lstm_state_dim)
    x_train.append(x)
    y_train.append(y)
    grid_train.append(g)
    zeros_train.append(z)
    print("\n", train_dataset_kind)
    print("\nx",x ,"\ny" , y,"\ng",g,"\nz", z)
x_train = np.concatenate(x_train, axis=0).astype(np.float32)
y_train = np.concatenate(y_train, axis=0).astype(np.float32)
grid_train = np.concatenate(grid_train, axis=0).astype(np.float32)
zeros_train = np.concatenate(zeros_train, axis=0).astype(np.float32)

train_data = (x_train, y_train, grid_train, zeros_train)
