from collections import OrderedDict
import numpy as np
import torch
from nerf.rays import get_rays
from nerf.nerf_maml_utils import render
from metrics import img2mse
from nerf.models.multiplane import ImagePlane

def update_params_sgd(loss, params, step_size=0.5, first_order=True, filter_params = None):
    name_list, tensor_list = zip(*params.items())
    grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order)
    updated_params = OrderedDict()
    for name, param, grad in zip(name_list, tensor_list, grads):
        if name in filter_params:
            updated_params[name] = param - step_size * grad
        else:
            updated_params[name] = param

    return updated_params


def inner_train_step_maml(model, support_data, args, filter_params):
    """Inner training step procedure."""
    # filter_params = layers2update(args)
    # obtain final prediction
    updated_params_fn = OrderedDict(model["network_fn"].named_parameters())
    if args.N_importance:
        updated_params_fine = OrderedDict(model["network_fine"].named_parameters())
    else:
        updated_params_fine = None
    model["updated_params"] = {
        "network_fn": updated_params_fn,
        "network_fine": updated_params_fine
    }
    h, w, focal, k = support_data["hwfk"]

    img_i = np.random.choice(list(range(support_data["images"].size(0))))
    target = support_data["images"][img_i]
    target = torch.Tensor(target)
    pose = support_data["poses"][img_i, :3, :4] 
    if args.multiplane:
        image_plane = ImagePlane(
            focal,
            support_data["poses"].cpu().numpy(),
            support_data["images"].cpu().numpy(),
            args.views,
        )

        model["network_fn"].image_plane = image_plane
        if args.N_importance:
            model["network_fine"].image_plane = image_plane

    if args.N_rand is not None:
        rays_o, rays_d = get_rays(
            h, w, k, torch.Tensor(pose)
        )  # (H, W, 3), (H, W, 3)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, h - 1, h),
                torch.linspace(0, w - 1, w),
            ),
            -1,
        )  # (H, W, 2)

    coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
    select_inds = np.random.choice(
        coords.shape[0], size=[args.N_rand], replace=False
    )  # (N_rand,)
    select_coords = coords[select_inds].long()  # (N_rand, 2)
    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    batch_rays = torch.stack([rays_o, rays_d], 0)
    label = target[select_coords[:, 0], select_coords[:, 1]]

    for _ in range(args.inner_step):
        rgb, disp, acc, extras = render(
            h,
            w,
            k,
            chunk=args.chunk,
            rays=batch_rays,
            verbose=False,
            retraw=True,
            **model,
        )

        loss = img2mse(rgb, label)
        trans = extras['raw'][...,-1]

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], label)
            loss = loss + img_loss0

        if args.inner_optimizer == "sgd":
            updated_params_fn = update_params_sgd(
                loss, updated_params_fn, step_size=args.inner_lr, first_order=True, filter_params = filter_params
            )
            if args.N_importance > 0:
                updated_params_fine = update_params_sgd(
                    loss, updated_params_fine, step_size=args.inner_lr, first_order=True, filter_params = filter_params
                )
        else:
            raise Exception(f"only sgd and adam optimizers are available. You provided: {args.inner_optimizer}")

        model["updated_params"] = {
            "network_fn": updated_params_fn,
            "network_fine": updated_params_fine
        }
    
    return {"network_fine": updated_params_fine, "network_fn": updated_params_fn}