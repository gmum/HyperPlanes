import os
import torch
import numpy as np
import random

def save_args(args):
    os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
    f = os.path.join(args.basedir, args.expname, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        f = os.path.join(args.basedir, args.expname, "config.txt")
        with open(f, "w") as file:
            file.write(open(args.config, "r").read())


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_ds_size(ds_name: str, default_size: tuple = (128, 128)):
    if ds_name in ("ShapeNet128x128", "ShapeNet200x200"):
        res = ds_name.replace("ShapeNet", "")
        w, h = res.split("x")
        near, far = 2., 6.
    elif ds_name == "SRNCars":
        w, h = default_size
        near, far = 0.8, 1.8
    elif ds_name == "DTU":
        w, h = 400, 300 
        near, far = 0.1, 5
        #near, far = 0.5, 3.5
    elif ds_name == "blender":
        w, h = 400, 400
        near, far = 2., 6.
    else: 
        raise ValueError("Unsupported ds_name. Check configuration file)")
    return int(w), int(h), near, far