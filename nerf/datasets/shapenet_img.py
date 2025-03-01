import json
import os

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = "data/original"


class Shapenet(Dataset):
    def __init__(self, data_path, items, device="cuda") -> None:
        super().__init__()
        self.data_path = data_path
        self.device = device
        self.items = [item for item in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item_filepath = self.items[idx]
        with open(os.path.join(item_filepath, "transforms.json"), "r") as fp:
            meta = json.load(fp)
        camera_angle_x = float(meta["camera_angle_x"])
        imgs, poses = [], []
        for idx in np.arange(len(meta["frames"])):
            frame = meta["frames"][idx]
            fname = os.path.join(
                item_filepath, os.path.basename(frame["file_path"]) + ".png"
            )
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame["transform_matrix"]))
        H, W = imgs[0].shape[:2]
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        imgs = imgs[..., :3] * imgs[..., -1:] + 1 - imgs[..., -1:]
        poses = np.array(poses).astype(np.float32)
        imgs = torch.from_numpy(imgs)
        poses = torch.from_numpy(poses)
        focal = torch.tensor(focal, device=self.device)
        return imgs, poses, [H, W, focal]


if __name__ == "__main__":
    imgs_dir = os.path.join(DATA_DIR, "02958343")
    split_file = os.path.join(DATA_DIR, "car_splits.json")
    with open(DATA_DIR, "r") as read_file:
        splits = json.load(read_file)
    test_exs = [os.path.join(imgs_dir, d) for d in sorted(splits["test"])]

    shapnet_dataset = Shapenet(data_path=imgs_dir, items=test_exs)
    dataloader = DataLoader(shapnet_dataset, batch_size=1)
    for bi, batch in enumerate(dataloader):
        imgs, poses, hwf = batch
        print(imgs.shape)
