import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class NeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
    ):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.render_network = RenderNetwork(
            D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs
        )

    def forward(self, x, updated_params):
        return self.render_network.forward(x, updated_params)

    def parameters(self):
        return self.render_network.parameters()

    def set_image_plane(self, ip):
        self.image_plane = ip



class RenderNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(RenderNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
    
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
                for i in range(D - 1)
            ]
        )

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, updated_params):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)   
        if updated_params is None:     
            updated_params = OrderedDict(self.named_parameters())

        h = input_pts

        for i, l in enumerate(self.pts_linears):
            weight_name = f"render_network.pts_linears.{i}.weight"
            bias_name = f"render_network.pts_linears.{i}.bias"
            weight = updated_params[weight_name]
            bias = updated_params[bias_name]
            h = torch.nn.functional.linear(h, weight=weight, bias=bias)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha_weight = updated_params['render_network.alpha_linear.weight']
            alpha_bias = updated_params['render_network.alpha_linear.bias']
            alpha = torch.nn.functional.linear(h, weight = alpha_weight, bias = alpha_bias)

            feature_weight = updated_params['render_network.feature_linear.weight']
            feature_bias = updated_params['render_network.feature_linear.bias']
            feature = torch.nn.functional.linear(h, weight = feature_weight, bias = feature_bias)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                weight_name = f"render_network.views_linears.{i}.weight"
                bias_name = f"render_network.views_linears.{i}.bias"
                weight = updated_params[weight_name]
                bias = updated_params[bias_name]
                h = torch.nn.functional.linear(h, weight=weight, bias=bias)
                h = F.relu(h)

            weight_name = "render_network.rgb_linear.weight"
            bias_name = "render_network.rgb_linear.bias"
            weight = updated_params[weight_name]
            bias = updated_params[bias_name]
            rgb = torch.nn.functional.linear(h, weight=weight, bias=bias)

            outputs = torch.cat([rgb, alpha], -1)
        else:
            weight_name = "render_network.output_linear.weight"
            bias_name = "render_network.output_linear.bias"
            weight = updated_params[weight_name]
            bias = updated_params[bias_name]
            outputs = torch.nn.functional.linear(h, weight=weight, bias=bias)

        return outputs    
