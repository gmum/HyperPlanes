from torch.utils.data import Dataset
import torch
import os
import json
import imageio
import numpy as np
from pathlib import Path


class ShapenetDataset(Dataset):
    def __init__(self, data_path, items) -> None:
        super().__init__()
        self.data_path = data_path
        self.items = [item for item in items]

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item_filepath = self.items[idx]
        # print(item_filepath)
        with open(os.path.join(item_filepath, 'transforms.json'), 'r') as fp:
            meta = json.load(fp)
        camera_angle_x = float(meta['camera_angle_x'])
        imgs, poses = [], []
        for idx in np.arange(len(meta['frames'])):
            frame = meta['frames'][idx]
            fname = os.path.join(item_filepath, os.path.basename(frame['file_path']) + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        H, W = imgs[0].shape[:2]
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        imgs = imgs[...,:3] * imgs[...,-1:] + 1-imgs[...,-1:]
        poses = np.array(poses).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        poses = torch.from_numpy(poses)
        focal = torch.tensor(focal, device="cuda")
        return imgs, poses, [H, W, focal]
    

def build_shapenet(image_set, dataset_root, splits_path, num_views):
    """
    Args:
        image_set: specifies whether to return "train", "val" or "test" dataset
        dataset_root: root path of the dataset
        splits_path: file path that specifies train, val and test split
        num_views: num of views to return from a single scene
    """
    root_path = Path(os.path.join(dataset_root, splits_path))
    splits_path = Path(os.path.join(dataset_root, f"{splits_path}.json"))
    with open(splits_path, "r") as splits_file:
        splits = json.load(splits_file)
    
    all_folders = [root_path.joinpath(foldername) for foldername in sorted(splits[image_set])]
    dataset = ShapenetDataset(root_path, all_folders)

    return dataset
