import json
from pathlib import Path
from datasets.shapenet import ShapeNet128x128, ShapeNet200x200
from datasets.multi_object import SRNDataset
import os


def build_shapenet(
    ds_name, image_set: str, dataset_root: str, sample_name: str, num_views=None
):
    """
    Args:
        image_set: specifies whether to return "train", "val" or "test" dataset
        dataset_root: root path of the dataset
        splits_path: file path that specifies train, val and test split
        num_views: num of views to return from a single scene
    """
    if ds_name == "ShapeNet128x128":
        root_path = Path(dataset_root)
        splits_path = Path(
            os.path.join(root_path, f"{sample_name.rstrip('s')}_splits.json")
        )  # cars
        with open(splits_path, "r") as splits_file:
            splits = json.load(splits_file)
        root_path = root_path / sample_name
        all_folders = [
            root_path.joinpath(foldername) for foldername in sorted(splits[image_set])
        ]
        dataset = ShapeNet128x128(all_folders, num_views)

    elif ds_name == "ShapeNet200x200":
        train_set = True if image_set == "train" else False
        dataset = ShapeNet200x200(
            root_dir=dataset_root, classes=[sample_name], train=train_set
        )
    elif ds_name == "SRNCars":
        dataset = SRNDataset(dataset_root, stage=image_set)
    else:
        raise ValueError("Unsupported ds_name. Check configuration file.")
    return dataset
