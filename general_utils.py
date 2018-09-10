import json
import os
from datetime import datetime
from enum import unique, Enum, auto
from typing import Union, List

out_dim = 5
# pxy = pedestrian, x pos, y pos
pxy_dim = 3


@unique
class DatasetKind(Enum):
    """Human-trajectory datasets used in the Social Model paper."""
    eth = auto()
    hotel = auto()
    zara1 = auto()
    zara2 = auto()
    ucy = auto()


# image size (width, height) for each dataset to compute grid
_image_size_dict = {
    DatasetKind.eth: [640, 480],
    DatasetKind.hotel: [720, 576],
    DatasetKind.zara1: [720, 576],
    DatasetKind.zara2: [720, 576],
    DatasetKind.ucy: [720, 576]
}

# relative path to data dir for each dataset
_rel_data_dir_dict = {
    DatasetKind.eth: "eth/univ",
    DatasetKind.hotel: "eth/hotel",
    DatasetKind.zara1: "ucy/zara/zara01",
    DatasetKind.zara2: "ucy/zara/zara02",
    DatasetKind.ucy: "ucy/univ"
}

#This function checks the presence of dataset we selected for the test_data and returns it
#If the asked dataset is not present, an exception is thrown
def _check_dataset_kind(dataset_kind: Union[DatasetKind, str]) -> DatasetKind:
    if isinstance(dataset_kind, DatasetKind):
        return dataset_kind

    if isinstance(dataset_kind, str) and hasattr(DatasetKind, dataset_kind):
        return DatasetKind[dataset_kind]

    raise ValueError("Unknown test_dataset_kind: {}".format(dataset_kind))


def get_image_size(dataset_kind: Union[DatasetKind, str]) -> List[int]:
    dataset_kind = _check_dataset_kind(dataset_kind)
    return _image_size_dict[dataset_kind]

#This function returns the complete folder path for the data we need in train data
#The data is fetched through a forloop which takes one by one the train_data_kinds and 
#Returns the data directory
#_rel_data_dir_dict is a dictionary which stores all the data directories last
#portion of the directory corresponding to different data
def get_data_dir(root: str, dataset_kind: Union[DatasetKind, str]) -> str:
    dataset_kind = _check_dataset_kind(dataset_kind)
    data_dir = os.path.join(root, _rel_data_dir_dict[dataset_kind])
    return data_dir


def now_to_str(format: str = "%Y%m%d%H%M%S") -> str:
    return datetime.now().strftime(format)


def load_json_file(json_file: str) -> dict:
    with open(json_file, "r") as f:
        return json.load(f)


def dump_json_file(obj, json_file: str) -> None:
    with open(json_file, "w") as f:
        json.dump(obj, f, indent=4, sort_keys=True)
