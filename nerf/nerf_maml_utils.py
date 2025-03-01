import os
import time
import tqdm
import imageio
import torch
from PIL import Image
from tqdm import tqdm
from metrics import img2mse, to8b, mse2psnr, calculate_ssim, calculate_lpips
import numpy as np
import torch.nn.functional as F
from nerf.rays import get_rays, ndc_rays, sample_pdf

from nerf.models.nerf import NeRF
from nerf.models.embedder import get_embedder
from nerf.models.multiplane_maml import MultiImageNeRFMAML

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def render_path(dataset_name, render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, logger=False):
        
    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    
    if dataset_name == "DTU":
        # unskip = [1, 2, 3]
        # skip = [x for x in range(len(render_poses)) if x not in unskip]
        skip = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39] + [25, 22, 28, 40, 44, 48, 0, 8, 13]
        unskip = [x for x in range(len(render_poses)) if x not in skip]
    elif dataset_name == "blender":
        skip = []
        unskip = [x for x in range(len(render_poses)) if x not in skip]
    else:
        #unskip = [10, 50, 100, 150, 190]
        #skip = [x for x in range(len(render_poses)) if x not in unskip]
        skip = []
        unskip = [x for x in range(len(render_poses)) if x not in skip]
    

    t = time.time()
    for i, c2w in tqdm(enumerate(render_poses)):
        if i in skip:
            continue
        t = time.time()
        rgb, disp, acc, _, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        
        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            os.makedirs(savedir, exist_ok=True)
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
        if logger and savedir:
            values = savedir.split("/")
            logger_epoch = values[2].split("epoch")[-1]
            sample_id = values[-1]
            gt_img, pred_img = to8b(gt_imgs[i].detach().cpu().numpy()), to8b(rgbs[-1])
            img = Image.fromarray(np.hstack((gt_img, pred_img)))
            logger[f"images/{logger_epoch}/{sample_id}"].append(img, step=i)
            logger[f"gt/{logger_epoch}/{sample_id}"].append(Image.fromarray(gt_img), step=i)
            logger[f"pred/{logger_epoch}/{sample_id}"].append(Image.fromarray(pred_img), step=i)

    
    rgbs = np.stack(rgbs, 0)
    metrics = {}
    gt_imgs = gt_imgs[unskip]
    if gt_imgs is not None:
        loss = img2mse(rgbs, gt_imgs)
        for rgb, gt_img in zip(rgbs, gt_imgs):
            single_psnr = mse2psnr(img2mse(rgb, gt_img))
            logger["metrics/single_psnr"].append(single_psnr)
        psnr = mse2psnr(loss)
        ssim = calculate_ssim(rgbs, gt_imgs, logger)
        lpips = calculate_lpips(rgbs, gt_imgs, logger)
        metrics = {
            "loss": loss,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
        }
        print(" CALCULATED PSNR FOR TESTSET")
        print(psnr.item())
        
    disps = np.stack(disps, 0)

    return rgbs, disps, metrics


def lossfun_off_reg(rgb, density, reg_range=10, wb_prior=False, wb_range=20):
    '''
    Computes the occlusion regularization loss.

    Args:
        rgb (torch.Tensor): The RGB rays/images.
        density (torch.Tensor): The current density map estimate.
        reg_range (int): The number of initial intervals to include in the regularization mask.
        wb_prior (bool): If True, a prior based on the assumption of white or black backgrounds is used.
        wb_range (int): The range of RGB values considered to be a white or black background.

    Returns:
        float: The mean occlusion loss within the specified regularization range and white/black background region.
    '''
    # Compute the mean RGB value over the last dimension
    rgb_mean = rgb.mean(-1)
    
    # Compute a mask for the white/black background region if using a prior
    if wb_prior:
        white_mask = torch.where(rgb_mean > 0.99, 1, 0) # A naive way to locate white background
        black_mask = torch.where(rgb_mean < 0.01, 1, 0) # A naive way to locate black background
        rgb_mask = (white_mask + black_mask) # White or black background
        rgb_mask = rgb_mask[:, wb_range:].fill_(0) # White or black background range
    else:
        rgb_mask = torch.zeros_like(rgb_mean)
    
    # Create a mask for the general regularization region
    if reg_range > 0:
        rgb_mask[:reg_range].fill_(1) # Penalize the points in reg_range close to the camera
    
    # Compute the density-weighted loss within the regularization and white/black background mask
    #return 0.01 * (density * rgb_mask).mean()
    return 0


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=device).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std
        noise = noise.to("cuda")

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
        
    #occlusion_loss = lossfun_off_reg(rgb_map, depth_map)
    occlusion_loss = None
    return rgb_map, disp_map, acc_map, weights, depth_map, occlusion_loss


def render_rays(
    ray_batch,
    network_fn,
    network_query_fn,
    N_samples,
    retraw=False,
    lindisp=False,
    perturb=0.0,
    N_importance=0,
    network_fine=None,
    white_bkgd=False,
    raw_noise_std=0.0,
    verbose=False,
    pytest=False,
    updated_params=None
):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    #     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, updated_params["network_fn"], network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, occlusion_loss = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
    )

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
    
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            N_importance,
            det=(perturb == 0.0),
            pytest=pytest,
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, updated_params["network_fine"], run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map, occlusion_loss = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest
        )

    ret = {"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map}
    if retraw:
        ret["raw"] = raw
    if N_importance > 0:
        ret["rgb0"] = rgb_map_0
        ret["disp0"] = disp_map_0
        ret["acc0"] = acc_map_0
        ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
            continue

    return ret, occlusion_loss


def render(
    H,
    W,
    K,
    chunk=1024 * 32,
    rays=None,
    c2w=None,
    ndc=True,
    near=0.0,
    far=1.0,
    use_viewdirs=False,
    c2w_staticcam=None,
    **kwargs,
):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays
    
    if use_viewdirs:   
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1.0, rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(
        rays_d[..., :1]
    )
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret, occlusion_loss = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ["rgb_map", "disp_map", "acc_map"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict] + [occlusion_loss]

def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret, occlusion_loss = render_rays(rays_flat[i : i + chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret, occlusion_loss

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs, updated_params):
        return torch.cat(
            [
                fn(inputs[i : i + chunk], updated_params)
                for i in range(0, inputs.shape[0], chunk)
            ],
            0,
        )

    return ret

def perform_pe(inputs, viewdirs, use_viewdirs, multires, multires_views, i_embed):

    embed_fn, input_ch = get_embedder(multires, i_embed)
    input_ch_views = 0
    embeddirs_fn = None
    if use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(multires_views, i_embed)

    embed_fn, input_ch = get_embedder(multires, i_embed)
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
    return embedded

def run_network(
    inputs, viewdirs, updated_params, fn, embed_fn, embeddirs_fn, embed_multi, netchunk=1024 * 64
):
    """Prepares inputs and applies network 'fn'."""
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
    outputs_flat = batchify(fn, netchunk)(embedded, updated_params)  # updated_params
    outputs = torch.reshape(
        outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
    )
    return outputs


def create_nerf(count, args):
    """Instantiate NeRF's MLP model."""
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    
    embed_multi, _ = get_embedder(args.multires, 0)
    
    #from nerf.models.embedder import MultiResHashGrid
    #embed_multi = MultiResHashGrid(3)
    #optimizer_hash = torch.optim.Adam(params=list(embed_multi.parameters()), lr=0.001)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    if args.nerf_model == "nerf":
        model = NeRF(D=args.netdepth, W=args.netwidth,
                    input_ch=3, output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    elif args.nerf_model == "multiplane":
        model = MultiImageNeRFMAML(None, count, input_ch_views, embed_multi).to(device) 
    else:
        raise ValueError("Wrong model")
    
    grad_vars = list(model.parameters())
    
    #optimizer_hash = torch.optim.Adam(params=list(embed_multi.parameters()), lr=0.001)
    optimizer_hash = None

    network_query_fn = lambda inputs, viewdirs, updated_params, network_fn: run_network(
        inputs,
        viewdirs,
        updated_params,
        network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk,
        embed_multi=embed_multi,
    )

    model_fine = None
    if args.N_importance > 0:
        if args.nerf_model == "nerf":
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        elif args.nerf_model == "multiplane":
            model_fine = MultiImageNeRFMAML(None, count, input_ch_views, embed_multi).to(device) 
        else:
            raise ValueError("Model error")
        grad_vars += list(model_fine.parameters())

    
    # Create NeRF optimizer 
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate)

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname)))
            if "tar" in f
        ]

    # TODO
    print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Load model
        model.load_state_dict(ckpt["network_fn_state_dict"])

    ##########################

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb": args.perturb,
        "N_importance": args.N_importance,
        "network_fine": model_fine,
        "N_samples": args.N_samples,
        "network_fn": model,
        "use_viewdirs": args.use_viewdirs,
        "white_bkgd": args.white_bkgd,
        "raw_noise_std": args.raw_noise_std
        }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_dir != "llff" or args.no_ndc:
        print("Not ndc!")
        render_kwargs_train["ndc"] = False
        render_kwargs_train["lindisp"] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, optimizer_hash
