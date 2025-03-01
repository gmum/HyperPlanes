from collections import OrderedDict
import torch
import torch.nn as nn
import copy
import math
import numpy as np
from collections import OrderedDict
from nerf.models.multiplane import ImagePlane, TriPlane
from nerf.rays import get_rays
from metrics import img2mse
from nerf.nerf_maml_utils import render, render_path, get_embedder
from fewshot.generate_updates import inner_train_step_maml # generate, layers2update
TESTSAVEDIR = "outputs"

def get_hn_params(hn_output, params, lmda, prev = 0):
    output_dict = {}
    for l_name, l_shape in params.items():
        if "weight" in l_name:
            curr = prev + l_shape[1]
            output_dict[l_name] = lmda * hn_output[:l_shape[0], prev:curr]
            # c_w += 1
        if "bias" in l_name:
            curr = prev
            output_dict[l_name] = lmda * hn_output[:l_shape[0], curr]
            prev += 1
        prev = curr
    return output_dict, prev

def inner_train_step(model, embedder, support_data, hn, epoch, args, filter_params, current = None):
    """ Inner training step procedure. """
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
    imgs = support_data["images"]
    imgs = imgs.permute(0, 3, 1, 2).cuda()
    lmda = min(epoch / args.lmda_steps, 1) if args.lmda_steps != 0 else 1
    #lmda = 0   ## FOR MAML ONLY ###
    to_concat = imgs[:args.img2embed]   
    if args.weight_update and lmda != 0:
        params = [param for param in model["network_fn"].parameters() if param.shape[0] == 256]
        params_fn = [tensor.view(-1, 1) if len(tensor.shape) == 1 else tensor for tensor in params]
        params_fn = torch.cat(params_fn, dim = 1).view(-1)
        number_of_layers = len(params_fn) // (args.H * args.W * 3)
        params_fn = params_fn[:number_of_layers*args.H*args.W*3]
        params_fn = params_fn.view(-1, 3, args.H, args.W)
        
        params = [param for param in model["network_fine"].parameters() if param.shape[0] == 256]
        params_fine = [tensor.view(-1, 1) if len(tensor.shape) == 1 else tensor for tensor in params]
        params_fine = torch.cat(params_fine, dim = 1).view(-1)
        params_fine = params_fine[:number_of_layers*args.H*args.W*3]
        params_fine = params_fine.view(-1, 3, args.H, args.W)
        
        to_concat = torch.cat((to_concat, params_fn, params_fine), dim = 0)
        
    if args.views_update and lmda != 0:
        embed_fn, input_ch = get_embedder(4, 0)
        views = embed_fn(support_data["poses"].view(imgs.size(0), -1))
        views = views.reshape(-1)
        reproduce = math.ceil(args.H*args.W*3 / len(views))
        views = views.repeat(reproduce)
        views = views[:args.H*args.W*3]
        views = views.view(1, 3, args.H, args.W)
        to_concat = torch.cat((to_concat, views), dim = 0)
    
    if lmda != 0:
        embeddings, planes, planesV2 = embedder(to_concat)
        hn_out = hn(embeddings)
        output_dict = {}
        output_dict["network_fn"], prev = get_hn_params(hn_out, filter_params, lmda)
        if args.N_importance > 0:
            output_dict["network_fine"], _ = get_hn_params(hn_out, filter_params, lmda, prev)
    
    updated_params = model["updated_params"]  


    if lmda == 1:
        if args.update_multiply:
            for param in updated_params["network_fn"].keys():
                if param in filter_params:
                    updated_params['network_fn'][param] = updated_params['network_fn'][param] * output_dict['network_fn'][param]

            if args.N_importance > 0:
                for param in updated_params["network_fine"].keys():
                    if param in filter_params:
                        updated_params['network_fine'][param] = updated_params['network_fine'][param] * output_dict['network_fine'][param] 
        else:
            
            for param in updated_params["network_fn"].keys():
                if param in filter_params:
                    updated_params['network_fn'][param] = updated_params['network_fn'][param] + output_dict['network_fn'][param]

            if args.N_importance > 0:
                for param in updated_params["network_fine"].keys():
                    if param in filter_params:
                        updated_params['network_fine'][param] = updated_params['network_fine'][param] + output_dict['network_fine'][param]


    elif lmda != 0:
        updated_params_maml = inner_train_step_maml(model, support_data, args, filter_params)

        for param in updated_params["network_fn"].keys():
            if param in filter_params:
                updated_params['network_fn'][param] = updated_params_maml['network_fn'][param] * (1 - lmda) + output_dict['network_fn'][param]

        if args.N_importance > 0:
            for param in updated_params["network_fine"].keys():
                if param in filter_params:
                    updated_params['network_fine'][param] = updated_params['network_fine'][param] + args.inner_lr * (updated_params_maml['network_fine'][param] - updated_params['network_fine'][param]) * (1 - lmda) + output_dict['network_fine'][param]

    elif lmda == 0:
        updated_params_maml = inner_train_step_maml(model, support_data, args, filter_params)
        for param in updated_params["network_fn"].keys():
            updated_params['network_fn'][param] = updated_params_maml['network_fn'][param]
        for param in updated_params["network_fine"].keys():
            updated_params['network_fine'][param] = updated_params_maml['network_fine'][param]
        
    model["updated_params"] = {
        "network_fn": updated_params["network_fn"],
        "network_fine": updated_params["network_fine"]
    }
    l2_hypernorm = 0
    # for key in output_dict["network_fn"].keys():
    #     l2_hypernorm += torch.sqrt(torch.sum(output_dict["network_fn"][key] ** 2))
    #     l2_hypernorm += torch.sqrt(torch.sum(output_dict["network_fine"][key] ** 2))
    return model["updated_params"], planes, planesV2[:1, :, :, :]


def sample_rays(args, data):
    img_i = np.random.choice(list(range(data["images"].size(0))))
    target = data["images"][img_i]
    pose = data["poses"][img_i, :3, :4]
    h, w, focal, k = data["hwfk"]

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
    target_s = target[select_coords[:, 0], select_coords[:, 1]]
    return batch_rays, target_s


class HyperMAML(nn.Module):
    def __init__(self, model: dict, args, embedder, hypernet, multiplane, logger=None):
        super().__init__()
        self.encoder = model
        self.embedder = embedder
        self.args = args
        self.logger = logger
        self.hn = hypernet
        self.args.multiplane = multiplane
        self.target_layers = self.hn.target_layers

    def forward(self, data_shot, data_query, epoch):            
        batch_rays, target_s = sample_rays(self.args, data_query)
        h, w, focal, k = data_shot["hwfk"]
        updated_params, planes, planesV2 = inner_train_step(model = self.encoder,
                                                        embedder = self.embedder,
                                                        support_data = data_shot,
                                                        hn = self.hn,
                                                        epoch = epoch,
                                                        args = self.args,
                                                        filter_params = self.target_layers,
                                                        current = None)
        if self.args.planes_generation == "encoder" and self.args.planes != 0:
            images = torch.cat((data_shot["images"], planes), axis = 0)
        elif self.args.planes_generation == "standalone" or self.args.planes == 0:
            images = data_shot['images']
        else:
            raise Exception("!!!")

        if self.args.multiplane:
            image_plane = TriPlane(
                focal=focal,
                poses=data_shot["poses"],
                images=planesV2,
                #count=self.args.views,
                count=data_shot["poses"].size(0),
                device="cuda:0",
            )
            # image_plane = ImagePlane(
            #     focal=focal,
            #     poses=data_shot["poses"],
            #     images=images,
            #     #count=self.args.views,
            #     count=data_shot["poses"].size(0),
            #     device="cuda:0",
            # )

            self.encoder["network_fn"].image_plane = image_plane
            if self.args.N_importance:
                self.encoder["network_fine"].image_plane = image_plane
        
        self.encoder["updated_params"] = updated_params
        rgb, _, _, _, occlusion_loss = render(
            h,
            w,
            k,
            chunk=data_query["chunks"],
            rays=batch_rays,
            verbose=False,
            retraw=True,
            **self.encoder,
        )

        return rgb, target_s, planes, occlusion_loss

    def forward_eval(self, data_shot, data_query, epoch, savedir=True):
        updated_params, planes, planesV2 = inner_train_step(self.encoder, self.embedder, data_shot, self.hn, epoch, self.args, self.target_layers)
        self.encoder["updated_params"] = updated_params

        h, w, focal, k = data_shot["hwfk"]

        if self.args.planes_generation == "encoder" and self.args.planes != 0:
            images = torch.cat((data_shot["images"], planes), axis = 0)
        elif self.args.planes_generation == "standalone" or self.args.planes == 0:
            images = data_shot['images']
        else:
            raise Exception("!!!")

        self.eval()
        
        with torch.no_grad():
            if self.args.multiplane:
                # image_plane = ImagePlane(
                #     focal,
                #     data_shot["poses"],
                #     images,
                #     count=data_shot["poses"].size(0),
                #     device="cuda:0",
                # )
                image_plane = TriPlane(
                    focal=focal,
                    poses=data_shot["poses"],
                    images=planesV2,
                    #count=self.args.views,
                    count=data_shot["poses"].size(0),
                    device="cuda:0",
                )
                self.encoder["network_fn"].image_plane = image_plane
                if self.args.N_importance:
                    self.encoder["network_fine"].image_plane = image_plane
            if savedir:
                savedir = data_query["filepath"]
            else:
                savedir = None
            self.encoder["perturb"] = False
            self.encoder["raw_noise_std"] = 0
            _, _, metrics = render_path(
                self.args.dataset_name,
                data_query["poses"],
                (h, w, focal),
                k,
                self.args.chunk,
                self.encoder,
                gt_imgs=data_query["images"],
                savedir=savedir,
                logger=self.logger,
            )
            self.encoder["perturb"] = self.args.perturb
            self.encoder["raw_noise_std"] = self.args.raw_noise_std
            return metrics
