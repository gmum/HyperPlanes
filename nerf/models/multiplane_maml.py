import torch
import os
from collections import OrderedDict
def get_freq_reg_mask(pos_enc_length, current_iter, total_reg_iter=35000,
                      max_visible=None, type='submission'):
    '''
    Returns a frequency mask for position encoding in NeRF.
    
    Args:
        pos_enc_length (int): Length of the position encoding.
        current_iter (int): Current iteration step.
        total_reg_iter (int): Total number of regularization iterations.
        max_visible (float, optional): Maximum visible range of the mask. Default is None. 
          For the demonstration study in the paper.
        
        Correspond to FreeNeRF paper:
          L: pos_enc_length
          t: current_iter
          T: total_iter
    
    Returns:
        torch.Tensor: Computed frequency or visibility mask.
    '''
    if max_visible is None:
        # default FreeNeRF
        if current_iter < total_reg_iter:
            freq_mask = torch.zeros(pos_enc_length)  # all invisible
            ptr = pos_enc_length / 3 * current_iter / total_reg_iter + 1 
            ptr = ptr if ptr < pos_enc_length / 3 else pos_enc_length / 3
            int_ptr = int(ptr)
            freq_mask[: int_ptr * 3] = 1.0 
            freq_mask[int_ptr * 3 : int_ptr * 3 + 3] = (ptr - int_ptr)  # assign the fractional part
            return torch.clamp(torch.tensor(freq_mask), 1e-8, 1-1e-8)  # for numerical stability
        else:
            return torch.ones(pos_enc_length)
    else:
        # For the ablation study that controls the maximum visible range of frequency spectrum
        freq_mask = torch.zeros(pos_enc_length)
        freq_mask[: int(pos_enc_length * max_visible)] = 1.0
        return torch.tensor(freq_mask)

class RenderNetworkMAML(torch.nn.Module):
    def __init__(self, input_size, dir_count):
        super().__init__()

        #self.input_size = 64
        self.input_size = 64
        #self.input_size = 64
    
        #self.conv1 = torch.nn.Conv1d(in_channels=self.input_size, out_channels=self.input_size, kernel_size=3, padding=1)
        #self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=self.input_size, kernel_size=3, padding=1)  
        
        self.layers_main = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
        )

        self.layers_main_2 = torch.nn.Sequential(
            torch.nn.Linear(512 + self.input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )

        self.layers_sigma = torch.nn.Sequential(
            torch.nn.Linear(256 + self.input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

        self.layers_rgb = torch.nn.Sequential(
            torch.nn.Linear(256 + self.input_size + dir_count, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3),
        )
    
    def forward(self, triplane_code, viewdir, updated_params = None):
        if updated_params is None:
            updated_params = OrderedDict(self.named_parameters())
        # TODO

        # x = triplane_code.permute(1, 0)
        # x = self.conv1(x)   
        # triplane_code = x.permute(1, 0) 

        base_name = "render_network.layers_main."
        x = torch.nn.functional.linear(triplane_code, weight=updated_params[f"{base_name}0.weight"], bias=updated_params[f"{base_name}0.bias"])
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.linear(x, weight=updated_params[f"{base_name}2.weight"], bias=updated_params[f"{base_name}2.bias"])
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.linear(x, weight=updated_params[f"{base_name}4.weight"], bias=updated_params[f"{base_name}4.bias"])
        x = torch.nn.functional.relu(x)
        x1 = torch.concat([x, triplane_code], dim=1)

        base_name = "render_network.layers_main_2."
        x = torch.nn.functional.linear(x1, weight=updated_params[f"{base_name}0.weight"], bias=updated_params[f"{base_name}0.bias"])
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.linear(x, weight=updated_params[f"{base_name}2.weight"], bias=updated_params[f"{base_name}2.bias"])
        x = torch.nn.functional.relu(x)

        xs = torch.concat([x, triplane_code], dim=1)
        
        base_name = "render_network.layers_sigma."
        xs = torch.nn.functional.linear(xs, weight=updated_params[f"{base_name}0.weight"], bias=updated_params[f"{base_name}0.bias"])
        xs = torch.nn.functional.relu(xs)
        xs = torch.nn.functional.linear(xs, weight=updated_params[f"{base_name}2.weight"], bias=updated_params[f"{base_name}2.bias"])
        sigma = xs

        base_name = "render_network.layers_rgb."
        x = torch.concat([x, triplane_code, viewdir], dim=1)
        x = torch.nn.functional.linear(x, weight=updated_params[f"{base_name}0.weight"], bias=updated_params[f"{base_name}0.bias"])
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.linear(x, weight=updated_params[f"{base_name}2.weight"], bias=updated_params[f"{base_name}2.bias"])
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.linear(x, weight=updated_params[f"{base_name}4.weight"], bias=updated_params[f"{base_name}4.bias"])
        rgb = x
        return torch.concat([rgb, sigma], dim=1)


class MultiImageNeRFMAML(torch.nn.Module):
    def __init__(self, image_plane, count, dir_count, embedder):
        super(MultiImageNeRFMAML, self).__init__()
        self.image_plane = image_plane
        self.render_network = RenderNetworkMAML(count, dir_count)
        self.embedder = embedder
        self.input_ch_views = dir_count

    def parameters(self):
        return self.render_network.parameters()

    def set_image_plane(self, ip):
        self.image_plane = ip

    def forward(self, x, updated_params):
        input_pts, input_views = torch.split(x, [3, self.input_ch_views], dim=-1)
        x = self.image_plane(input_pts)
        
        triplane = True
        if triplane:
            x = x.mean(1)
            x = x.squeeze(0)
        
        # import pdb 
        # pdb.set_trace()
        #print(self.embedder(input_pts).size(-1)) 
        
        # ci = int(os.environ.get("epoch"))
        # mask = get_freq_reg_mask(123, ci).cuda()
        # pos_enc = self.embedder(input_pts)
        # pos_enc = mask * pos_enc
        # x = torch.cat((x, pos_enc), dim = 1)
        
        x = torch.cat((x, self.embedder(input_pts)[:, :32]), dim = 1)

        
        #x = torch.cat((x, input_pts), dim = 1)
        return self.render_network.forward(x, input_views, updated_params)
