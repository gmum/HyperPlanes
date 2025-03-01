import math
import os
from PIL import Image
import neptune
import numpy as np
import torch
import random
from ap import auxilary_poses
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from pixelnerf_data_utils.dataset import get_split_dataset
from config_parser import config_parser
from datasets import build_shapenet
from fewshot.hypermaml import HyperMAML
from hypernetwork import HN, calculate_hypernetwork_output
from metrics import calculate_ssim, img2mse, mse2psnr
from nerf.nerf_maml_utils import create_nerf
from utils import get_ds_size, save_args

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEBUG = False

def rgba2rgb(rgba, background=(255,255,255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )


CH = 3


layers2update = [
    "render_network.layers_rgb.4.weight",
    "render_network.layers_rgb.4.bias",
    "render_network.layers_sigma.2.weight",
    "render_network.layers_sigma.2.bias",
    "render_network.layers_main_2.2.weight",
    "render_network.layers_main_2.2.bias",
]


parser = config_parser()
args = parser.parse_args()
os.environ["multires"] = str(args.multires)
basedir = args.basedir
expname = args.expname


W, H, NEAR, FAR = get_ds_size(args.dataset_name)
args.W = W
args.H = H

bds_dict = {
    "near": NEAR,
    "far": FAR,
}

# Save args
save_args(args)

# Neptune logger
run = None
if args.neptune:
    run = neptune.init_run(
    project="hypermaml/MultiPlaneNeRF-MAML",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhNzEyZGQ3Yi0yNDJkLTQ3ODgtOGYxYy0xM2ZlN2RiMWI0NDAifQ==",
)
    run["parameters"] = args

###########
# DATASET #
###########

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
        i_test = np.random.choice(test_idx, size=25, replace=False)
    elif args.views in [4, 8]:
        i_train = np.random.choice(train_idx, size=args.views, replace=False)
        if run:
            run["i_train"] = str(i_train)
        i_test = test_idx
    
    train_images, train_poses = torch.tensor(np.array(images, dtype=np.float32)[i_train]), torch.tensor(np.array(poses)[i_train])
    test_images, test_poses = torch.tensor(np.array(images, dtype=np.float32)[test_idx]), torch.tensor(np.array(poses)[test_idx])
    
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


##################
# Target Network #
##################
np.random.seed(0)
render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, optimizer_hash = create_nerf(
    args.views + args.planes, args
)

# Setup NEAR and FAR params
render_kwargs_train.update(bds_dict)
render_kwargs_test.update(bds_dict)


################
# HyperNetwork #
################

o_h, output_dim, layers = calculate_hypernetwork_output(
    layers2update, render_kwargs_train["network_fn"]
)

output_dim = output_dim * 2 if args.N_importance > 0 else output_dim

def initialize_plane(shape) -> torch.Tensor:
    initialized_tensor = torch.empty(shape, device="cuda:0", requires_grad=True)
    torch.nn.init.kaiming_uniform_(initialized_tensor, a=0, mode='fan_in', nonlinearity='relu')
    return initialized_tensor.detach().requires_grad_()

def initialize_auxilary_planes(num_planes, plane_shape) -> torch.Tensor:
    planes = []
    for _ in range(num_planes):
        plane = initialize_plane(plane_shape)
        planes.append(plane.detach().requires_grad_())
    auxilary_planes = torch.stack(planes, dim=0)
    return auxilary_planes.detach().requires_grad_()

if args.planes != 0:
    auxilary_poses = auxilary_poses[:args.planes, :, :]
    if args.planes_generation == "standalone":
        auxilary_planes = initialize_auxilary_planes(args.planes, (128, 128, 3))
        params2optimize = [torch.nn.Parameter(plane) for plane in auxilary_planes]
        optimizer_planes = torch.optim.Adam(params=params2optimize, lr=args.planes_lr)  

# if args.planes != 0:
#     auxilary_poses = auxilary_poses[:args.planes, :, :]
#     if args.planes_generation == "standalone":
#         auxilary_planes = initialize_tensors((args.planes, 128, 128, 3))
#         auxilary_planes_flat = auxilary_planes.view(args.planes, -1)
#         params2optimize = [torch.nn.Parameter(auxilary_planes_flat)]
#         optimizer_planes = torch.optim.Adam(params=params2optimize, lr=args.planes_lr)  

    
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
    number_of_layers = len(params_fn) // (H * W * CH)
    embedding_size += 2 * number_of_layers
if args.views_update:
    embedding_size += 1

hypernet = HN(args.hm_hn_len, args.hm_hn_width, output_dim, layers, embedding_size)

############
# Encoders #
############

if args.backbone_class == "Conv4":
    from encoders.convnet import ConvNet
    if args.planes_generation == "standalone":
        embedder = ConvNet(z_dim=o_h)
    elif args.planes_generation == "encoder":
        embedder = ConvNet(z_dim=o_h, planes=args.planes)
elif args.backbone_class == "ResNet" and args.dataset_name != "ShapeNet128x128":
    from encoders.resnet import ResNet101

    embedder = ResNet101()
else:
    raise Exception(
        f"You can only specify Conv4 as an encoder. You specified {args.backbone_class}"
    )


# Optimizers
optimizer_embedder = torch.optim.Adam(params=embedder.parameters(), lr=args.encoder_rate)
optimizer_hypernet = torch.optim.Adam(params=hypernet.parameters(), lr=args.hn_rate)


is_multiplane = True if args.nerf_model == "multiplane" else False
model = HyperMAML(
    render_kwargs_train, args, embedder, hypernet, multiplane=is_multiplane, logger=run
).to(device)

if os.path.exists(args.checkpoint):
    checkpoint = torch.load(args.checkpoint)
    model.encoder["network_fn"].load_state_dict(checkpoint["network_fn_state_dict"])
    model.encoder["network_fine"].load_state_dict(checkpoint["network_fine_state_dict"])
    hypernet.load_state_dict(checkpoint["hn"])
    embedder.load_state_dict(checkpoint["embedder"])

#############
# MAIN LOOP #
#############
global_step = start
N_iters = args.i_epochs + 1
print("Begin")

start = start + 1

epoch = 0 
for i in trange(start, N_iters):
    metrics_storage = {"loss": [], "psnr": [], "ssim": [], "lpips": []}
    mean_metrics = {}
    model.train()
    model.zero_grad()
    os.environ["epoch"] = str(epoch)

    losses = 0
    psnrs = 0
    ssims = 0
    hypernorms = 0
    for bi, el in tqdm(enumerate(train_loader)):
        # if bi > 2:
        #     break
        losses_batch = 0
        psnrs_batch = []
        hypernorms_batch = 0
        ssims_batch = 0
        range2iterate = el["images"].size(0) if args.dataset_name not in ["DTU", "blender"] else 1
        for j in range(range2iterate):
            if args.dataset_name == "DTU":
                imgs, poses = el["images"].to(device), el["poses"].to(device)
                imgs = imgs.permute(0, 2, 3, 1)
                imgs = imgs * 0.5 + 0.5
                focal = el["focal"][1]
                #imgs = imgs / 400
            elif args.dataset_name == "blender":
                imgs = train_loader["images"].to(device)
                poses = train_loader['poses'].to(device)
                focal = train_loader["hwf"][-1]
            else:
                imgs, poses = el["images"][j, :, :, :].to(device), el["poses"][j, :, :].to(device)

                hwf = el["hwf"][j].to(device)
                _, _, focal = hwf
            
            h, w = args.H, args.W
            k = np.array(
                [[float(focal), 0, 0.5 * w], [0, float(focal), 0.5 * h], [0, 0, 1]]
            )
            hwfk = (h, w, focal, k)

            ################ ALWAYS THE SAME VIEWING DIRECTION IN SUPPORT SET ################
            if args.dataset_name not in ["blender", "DTU"]:
                support_images, query_images = torch.split(imgs, [args.views, args.views_query], dim=0)
                support_poses, query_poses = torch.split(poses, [args.views, args.views_query], dim=0)
            elif args.dataset_name == "DTU":
                if args.views == 1:
                    train_idx = [25]
                elif args.views == 3:
                    train_idx = [22, 25, 28]
                elif args.views == 6:
                    train_idx = [22, 25, 28, 40, 44, 48]
                elif args.views == 9:
                    train_idx = [22, 25, 28, 40, 44, 48, 0, 8, 13]
                test_idx = [x for x in range(imgs.size(0)) if x not in train_idx]
                support_images, query_images = imgs[train_idx], imgs[test_idx] 
                support_poses, query_poses = poses[train_idx], poses[test_idx]
            elif args.dataset_name == "blender":
                support_images, query_images = imgs, imgs 
                support_poses, query_poses = poses, poses
            ################ ALWAYS THE SAME VIEWING DIRECTION IN SUPPORT SET ################
            
            ################ RANDOM VIEWING DIRECTION IN SUPPORT SET ################

            # random_idx = random.sample(range(imgs.size(0)), args.views)
            # query_idx = [idx for idx in range(imgs.size(0)) if idx not in random_idx]
            # support_images, support_poses = imgs[random_idx], poses[random_idx]
            # query_images, query_poses = imgs[query_idx], poses[query_idx]
            ################ RANDOM VIEWING DIRECTION IN SUPPORT SET ################

            if args.planes != 0:
                if args.planes_generation == "standalone":
                    support_images = torch.cat((support_images, auxilary_planes), dim = 0)
                support_poses = torch.cat((support_poses, auxilary_poses), dim = 0)
                
            support_data = {
                "images": support_images,
                "poses": support_poses,
                "hwfk": hwfk,
            }

            query_data = {
                "images": query_images if args.views_query > 0 else support_images,
                "poses": query_poses if args.views_query > 0 else support_poses,
                "hwfk": hwfk,
                "chunks": args.chunk,
            }

            if args.planes_generation == "encoder" and args.planes != 0:
                logits, label, auxilary_planes, occlusion_loss = model(support_data, query_data, epoch)
            elif args.planes_generation == "standalone" or args.planes == 0:
                logits, label, _, occlusion_loss = model(support_data, query_data, epoch)
            else:
                raise Exception("!!!")
            img_loss = img2mse(logits, label)
            loss = img_loss
            if occlusion_loss:
                loss + occlusion_loss
            if torch.isinf(loss):
                raise Exception(
                    f"loss is to small. You need to decrease hyper_constant. Your hyper constant now: {args.hyper_constant}"
                )

            ssim = calculate_ssim(logits, label, None)
            psnr = mse2psnr(img_loss)
            losses_batch += loss / imgs.size(0)
            ssims_batch += ssim / imgs.size(0)
            hypernorms_batch += 0
            epoch += 1
    
            if psnr.item() != math.inf and not np.isnan(psnr.item()):
                psnrs_batch.append(psnr.item())
            else:
                continue

        losses += losses_batch.item() / len(train_loader)
        psnrs += np.mean(psnrs_batch) / len(train_loader)
        if np.isnan(psnrs):
            continue
        else:
            print(f"LOSS: {round(loss.item(), 2)}; PSNR: {round(psnrs, 2)}")
        ssims += ssim.item() / len(train_loader)
        hypernorms += 0

        optimizer.zero_grad()
        optimizer_hypernet.zero_grad()
        optimizer_embedder.zero_grad()
        #optimizer_hash.zero_grad()
        if args.planes != 0 and args.trainable_planes and args.planes_generation == "standalone":
            optimizer_planes.zero_grad()
        losses_batch.backward()
        optimizer.step()
        #optimizer_hash.step()
        optimizer_hypernet.step()
        optimizer_embedder.step()
        if args.planes != 0 and args.trainable_planes and args.planes_generation == "standalone":
            optimizer_planes.step()   
        # if torch.sum(auxilary_planes.grad) == 0:
        #     raise ValueError("No gradient flow through planes") 
        # else:
        if args.dataset_name == "blender":
            print(bi)
            break
        if run and args.planes != 0 and args.trainable_planes and args.planes_generation == "standalone":
            run["auxilary_planes_gradients"].append(torch.sum(auxilary_planes.grad))

    if np.isnan(psnrs):
        raise Exception("PSNR is NAN")
    if run:
        run[f"metrics/psnr"].append(psnrs)
        run[f"metrics/loss"].append(losses)
        run[f"metrics/ssim"].append(ssims)
        run[f"metrics/hypernorm"].append(hypernorms)

    print(f"TRAIN | Epoch: {i} PSNR: {psnrs}")
    global_step += 1

    ##############
    # CHECKPOINT #
    ##############
    if i % args.i_weights == 0 or i == args.i_epochs or i == 30:
        path = os.path.join(basedir, expname, "{:06d}.pt".format(i))
        network_fine_state = (
            render_kwargs_train["network_fine"].state_dict()
            if args.N_importance > 0
            else None
        )
        torch.save(
            {
                "global_step": global_step,
                "network_fn_state_dict": render_kwargs_train["network_fn"].state_dict(),
                "network_fine_state_dict": network_fine_state,
                "nerf_optimizer": optimizer.state_dict(),
                "hn": hypernet.state_dict(),
                "hn_optimizer": optimizer_hypernet.state_dict(),
                "embedder": embedder.state_dict(),
                "embedder_optimizer": optimizer_embedder.state_dict(),
                "hypermaml": model.state_dict(),
            },
            path,
        )
        print("Saved checkpoints at", path)
        if run:
            run[f"model/epoch_{i}.pt"].upload(path)

    small_test_condition = ((i % 10000 == 0) and i != 0)
    
    #small_test_condition = True
    if args.planes != 0 and (i % args.i_auxilary_planes == 0 or small_test_condition):
        if run:
            for idx, auxilary_plane in enumerate(auxilary_planes):
                arr = auxilary_plane.detach().cpu().numpy() * 255
                arr = arr.astype(np.uint8)
                img = Image.fromarray(arr)
                run[f"auxilary_planes/{i}"].append(img, step=idx)

    if (i % args.i_testset == 0) or small_test_condition:
        metrics_storage = {"loss": [], "psnr": [], "ssim": [], "lpips": []}
        for bi_test, el in enumerate(tqdm(test_loader)):
            range2iterate = el["images"].size(0) if args.dataset_name not in ["DTU", "blender"] else 1
            if args.dataset_name == "DTU":
                imgs, poses = el["images"].to(device), el["poses"].to(device)
                imgs = imgs.permute(0, 2, 3, 1)
                imgs = imgs * 0.5 + 0.5
                focal = el["focal"][1]
                #imgs = imgs / 400
            elif args.dataset_name == "blender":
                imgs_train = train_loader["images"].to(device)
                poses_train = train_loader['poses'].to(device)

                imgs_test = test_loader["images"].to(device)
                poses_test = test_loader['poses'].to(device)
                focal = train_loader["hwf"][-1]

            else:
                imgs, poses = el["images"][j, :, :, :].to(device), el["poses"][j, :, :].to(device)

                hwf = el["hwf"][j].to(device)
                _, _, focal = hwf
            
            ################ ALWAYS THE SAME VIEWING DIRECTION IN SUPPORT SET ################
            if args.dataset_name not in ["blender", "DTU"]:
                support_images, query_images = torch.split(imgs, [args.views, args.views_query], dim=0)
                support_poses, query_poses = torch.split(poses, [args.views, args.views_query], dim=0)
            elif args.dataset_name == "DTU":
                if args.views == 1:
                    train_idx = [25]
                elif args.views == 3:
                    train_idx = [22, 25, 28]
                elif args.views == 6:
                    train_idx = [22, 25, 28, 40, 44, 48]
                elif args.views == 9:
                    train_idx = [22, 25, 28, 40, 44, 48, 0, 8, 13]
                test_idx = [x for x in range(imgs.size(0)) if x not in train_idx]
                support_images, query_images = imgs[train_idx], imgs[test_idx] 
                support_poses, query_poses = poses[train_idx], poses[test_idx]
            elif args.dataset_name == "blender":
                support_images, query_images = imgs_train, imgs_test 
                support_poses, query_poses = poses_train, poses_test
                
            ################ ALWAYS THE SAME VIEWING DIRECTION IN SUPPORT SET ################
            
            ################ RANDOM VIEWING DIRECTION IN SUPPORT SET ################
            
            # support_images, support_poses = imgs[random_idx].unsqueeze(0), poses[random_idx].unsqueeze(0)
            # query_images, query_poses = imgs[query_idx], poses[query_idx]
            ################ RANDOM VIEWING DIRECTION IN SUPPORT SET ################
             
                        
            h, w = args.H, args.W
            k = np.array(
                [[float(focal), 0, 0.5 * w], [0, float(focal), 0.5 * h], [0, 0, 1]]
            )
            hwfk = (h, w, focal, k)

            if args.planes != 0:
                if args.planes_generation == "standalone":
                    support_images = torch.cat((support_images, auxilary_planes), dim = 0)
                support_poses = torch.cat((support_poses, auxilary_poses), dim = 0)

            support_data = {
                "images": support_images,
                "poses": support_poses,
                "hwfk": hwfk,
            }

            query_data = {
                "images": query_images if args.views_query > 0 else support_images,
                "poses": query_poses if args.views_query > 0 else support_poses,
                "hwfk": hwfk,
                "chunks": args.chunk,
                "filepath": f"outputs/{args.expname}/epoch{i}/{bi_test}",
            }

            metrics = model.forward_eval(
                support_data, query_data, ((i + bi) / len(train_loader))
            )
            for key in metrics:
                metrics_storage[key].append(metrics[key].item())
            
            if args.dataset_name == "blender":
                break
            elif args.dataset_name == "DTU":
                break

        if run:
            for metric_name in metrics_storage:
                run[f"metrics/test_{metric_name}"].append(
                    np.mean(metrics_storage[metric_name])
                )
