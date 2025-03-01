import os
import numpy as np
import neptune
import torch
from pixelnerf_data_utils.dataset import get_split_dataset
from tqdm import tqdm
from datasets import build_shapenet
from torch.utils.data import DataLoader
from metrics import img2mse
from metrics import img2mse, to8b, mse2psnr, calculate_ssim, calculate_lpips
from nerf.nerf_maml_utils import create_nerf

from config_parser import config_parser
from utils import save_args, set_seed, get_ds_size
from hypernetwork import HN, calculate_hypernetwork_output
from fewshot.hypermaml import HyperMAML

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

parser = config_parser()
args = parser.parse_args()

basedir = args.basedir
expname = args.expname

W, H, NEAR, FAR = get_ds_size(args.dataset_name)
args.W = W
args.H = H

bds_dict = {
    "near": NEAR,
    "far": FAR,
}



layers2update = [
    "render_network.layers_rgb.4.weight",
    "render_network.layers_rgb.4.bias",
    "render_network.layers_sigma.2.weight",
    "render_network.layers_sigma.2.bias",
    "render_network.layers_main_2.2.weight",
    "render_network.layers_main_2.2.bias",
]

run = None
if args.neptune:
    run = neptune.init_run(
    project="hypermaml/MultiPlaneNeRF-MAML",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhNzEyZGQ3Yi0yNDJkLTQ3ODgtOGYxYy0xM2ZlN2RiMWI0NDAifQ==",
)
    run["parameters"] = args

save_args(args)
set_seed()


if args.dataset_name == "DTU":
    train_loader, test_loader , _ = get_split_dataset(dataset_type="dvr_dtu", datadir=args.dataset_dir)
elif args.dataset_name == "blender":
    from datasets.blender import load_blender_data
    images = []
    poses = []
    loader = load_blender_data(args.dataset_dir)
    for img, pose in tqdm(zip(loader["imgs"], loader["poses"])):
        #images.append(rgba2rgb(img))
        if args.white_bkgd:
            img_ = img[...,:3]*img[...,-1:] + (1.-img[...,-1:])
        else:
            img_ = img
        images.append(img_)
        poses.append(pose)

    train_idx = loader["i_split"][0]
    test_idx = loader["i_split"][-1]
    
    if args.views == 1:
        i_train = [28]
    elif args.views == 14:
        i_train = [2, 5, 8, 9, 10, 16, 34, 35, 40, 52, 53, 54, 58, 60]
    elif args.views in [4, 8]:
        i_train = np.random.choice(train_idx, size=args.views, replace=False)
        i_test = test_idx
    
    run["i_train"] = i_train
    
    #i_test = np.random.choice(test_idx, size=35, replace=False) 
    i_test = test_idx
    # i_test = [113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
    #           125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137]
    
    train_images, train_poses = torch.tensor(np.array(images, dtype=np.float32)[i_train]), torch.tensor(np.array(poses)[i_train])
    test_images, test_poses = torch.tensor(np.array(images, dtype=np.float32)[i_test]), torch.tensor(np.array(poses)[i_test])
    
    train_loader = {"images": train_images, "poses": train_poses, "hwf": loader["hwf"]}

    test_loader = {"images": test_images, "poses": test_poses, "hwf": loader["hwf"]}
else:
    dataset = build_shapenet(
        args.dataset_name,
        "train",
        args.dataset_dir,
        sample_name=args.dataset_sample,
        num_views=args.views + args.views_query,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        generator=torch.Generator(device="cpu"),
    )

    dataset_test = build_shapenet(
        args.dataset_name,
        "test",
        args.dataset_dir,
        sample_name=args.dataset_sample,
        num_views=args.views + args.views_query,
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        generator=torch.Generator(device="cpu"),
    )


# Create nerf model
render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, optimizer_hash = create_nerf(
    args.views, args
)
global_step = start


render_kwargs_train.update(bds_dict)
render_kwargs_test.update(bds_dict)


o_h, output_dim, layers = calculate_hypernetwork_output(
    layers2update, render_kwargs_train["network_fn"]
)  # network_fn == network_fine

output_dim = output_dim * 2 if args.N_importance > 0 else output_dim

embedding_size = args.img2embed
if args.weight_update:
    params = [
        param
        for param in render_kwargs_train["network_fn"].parameters()
        if param.shape[0] == 256
    ]
    params_fn = [
        tensor.view(-1, 1) if len(tensor.shape) == 1 else tensor for tensor in params
    ]
    params_fn = torch.cat(params_fn, dim=1).view(-1)
    number_of_layers = len(params_fn) // (args.H * args.W * 3)
    embedding_size += 2 * number_of_layers
if args.views_update:
    embedding_size += 1

hypernet = HN(args.hm_hn_len, args.hm_hn_width, output_dim, layers, embedding_size)

# Prepare HyperMAML modules

if args.backbone_class == "Conv4":
    from encoders.convnet import ConvNet

    embedder = ConvNet(z_dim=o_h)
elif args.backbone_class == "ResNet" and args.dataset_name != "ShapeNet128x128":
    from encoders.resnet import ResNet101

    embedder = ResNet101()
else:
    raise Exception(
        f"You can only specify Conv4 as an encoder. You specified {args.backbone_class}"
    )



N_iters = args.i_epochs + 1
print("Begin")

start = start + 1
is_multiplane = True if args.nerf_model == "multiplane" else False

checkpoint = torch.load(args.checkpoint)


metrics_storage = {"loss": [], "psnr": [], "ssim": [], "lpips": []}
i = 1001
os.makedirs(f"logs/{args.expname}/epoch{i}", exist_ok=True)

psnr_mean = []
epoch = 0
os.environ["epoch"] = "0"
for bi_test, el in enumerate(tqdm(test_loader)):
    #eval_max = 200 if args.dataset_name == "ShapeNet200x200" else -1
    # if bi_test > eval_max:
    #     break
    # model = deepcopy(model_initial)
    model = HyperMAML(
        render_kwargs_train,
        args,
        embedder,
        hypernet,
        multiplane=is_multiplane,
        logger=run,
    ).to(device)


    model.encoder["network_fn"].load_state_dict(checkpoint["network_fn_state_dict"], strict=False)
    model.encoder["network_fine"].load_state_dict(checkpoint["network_fine_state_dict"], strict=False)
    hypernet.load_state_dict(checkpoint["hn"])
    embedder.load_state_dict(checkpoint["embedder"])
    
    optimizer_embedder = torch.optim.Adam(
        params=embedder.parameters(), lr=args.encoder_rate
    )
    optimizer_hypernet = torch.optim.Adam(params=hypernet.parameters(), lr=args.hn_rate)

    if args.dataset_name == "DTU":
        imgs, poses = el["images"].to(device), el["poses"].to(device)
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = imgs * 0.5 + 0.5
        focal = el["focal"][1]
    elif args.dataset_name == "blender":
        imgs_train = train_loader["images"].to(device)
        poses_train = train_loader['poses'].to(device)

        imgs_test = test_loader["images"].to(device)
        poses_test = test_loader['poses'].to(device)
        focal = train_loader["hwf"][-1]
    else:
        imgs, poses = el["images"].to(device), el["poses"].to(device)
        imgs = imgs.squeeze(0)
        poses = poses.squeeze(0)
        hwf = el["hwf"].to(device)
        h, w, focal = hwf[0][0], hwf[0][1], hwf[0][2]

    if args.dataset_name == "DTU":
        if args.views == 1:
            train_idx = [25]
        elif args.views == 3:
            train_idx = [22, 25, 28]
        elif args.views == 6:
            train_idx = [22, 25, 28, 40, 44, 48]
        elif args.views == 9:
            train_idx = [22, 25, 28, 40, 44, 48, 0, 8, 13]
            
        test_idx = [x for x in range(49) if x not in train_idx]
        
        support_images, query_images = imgs[train_idx], imgs[test_idx]
        support_poses, query_poses = poses[train_idx], poses[test_idx]
    if args.dataset_name == "SRNCars":
        train_idx = [64, 104]
        test_idx = [x for x in range(imgs.size(0)) if x not in train_idx]
        
        
            
        support_images, query_images = imgs[train_idx], imgs[test_idx]
        support_poses, query_poses = poses[train_idx], poses[test_idx]
    
    if args.dataset_name == "blender":
        support_images, query_images = imgs_train, imgs_test
        support_poses, query_poses = poses_train, poses_test
    
                        
    h, w = args.H, args.W
    k = np.array(
        [[float(focal), 0, 0.5 * w], [0, float(focal), 0.5 * h], [0, 0, 1]]
    )
    hwfk = (h, w, focal, k)

    support_data = {
        "images": support_images,
        "poses": support_poses,
        "hwfk": hwfk,
        "chunks": args.chunk,
        "filepath": f"logs/{args.expname}/epoch{i}/{bi_test}",
    }

    query_data = {
        "images": query_images,
        "poses": query_poses,
        "hwfk": hwfk,
        "chunks": args.chunk,
        "filepath": f"logs/{args.expname}/epoch{i}/{bi_test}",
    }
    for _ in tqdm(range(args.eval_pretraining_iters)):
        logits, label, _, occlusion_loss = model(
            support_data, support_data, ((i + bi_test) / len(train_loader))
        )
        loss = img2mse(logits, label) 
        psnr = mse2psnr(loss)
        if occlusion_loss:
            loss = loss + occlusion_loss
        if run:
            run[f"pretrening_losses/{bi_test}"].append(loss)
            run[f"pretrening_psnrs/{bi_test}"].append(psnr)

        print(psnr)
        optimizer_hypernet.zero_grad()
        optimizer_embedder.zero_grad()
        optimizer.zero_grad()
        #optimizer_hash.zero_grad()
        loss.backward()
        optimizer.step()
        #optimizer_hash.step()
        optimizer_hypernet.step()
        optimizer_embedder.step()
        epoch += 1
        os.environ["epoch"] = str(epoch)

    metrics = model.forward_eval(
        support_data, query_data, ((i + bi_test) / len(train_loader))
    )
    # psnr_mean.append(metrics.item())
    if run:
        run["metrics/psnr"].append(metrics["psnr"].item())
        run["metrics/ssim"].append(metrics["ssim"].item())
        run["metrics/lpips"].append(metrics["lpips"].item())
    for key in metrics:
        metrics_storage[key].append(metrics[key].item())
    
    if args.dataset_name == "blender":
        break
if run:
    for metric_name in metrics_storage:
        run[f"metrics/final_{metric_name}"].append(
            np.mean(metrics_storage[metric_name])
        )
