import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None


    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out
    
class RenderNetworkFW(torch.nn.Module):
    def __init__(self, input_size, dir_count):
        super().__init__()
        self.input_size = 3 * input_size + input_size * 3

        self.layers_main = torch.nn.Sequential(
            Linear_fw(self.input_size, 2048),
            torch.nn.ReLU(),
            Linear_fw(2048, 512),
            torch.nn.ReLU(),
            Linear_fw(512, 512),
            torch.nn.ReLU(),
        )

        self.layers_main_2 = torch.nn.Sequential(
            Linear_fw(512 + self.input_size, 256),
            torch.nn.ReLU(),
            Linear_fw(256, 256),
            torch.nn.ReLU(),
        )

        self.layers_sigma = torch.nn.Sequential(
            Linear_fw(256 + self.input_size, 256),
            torch.nn.ReLU(),
            Linear_fw(256, 1),
        )

        self.layers_rgb = torch.nn.Sequential(
            Linear_fw(256 + self.input_size + dir_count, 256),
            torch.nn.ReLU(),
            Linear_fw(256, 256),
            torch.nn.ReLU(),
            Linear_fw(256, 3),
        )

    def forward(self, triplane_code, viewdir):

        x = self.layers_main(triplane_code)
        x1 = torch.concat([x, triplane_code], dim=1)

        x = self.layers_main_2(x1)
        xs = torch.concat([x, triplane_code], dim=1)

        sigma = self.layers_sigma(xs)
        x = torch.concat([x, triplane_code, viewdir], dim=1)
        rgb = self.layers_rgb(x)
        return torch.concat([rgb, sigma], dim=1)


class MultiImageNeRF(torch.nn.Module):
    def __init__(self, image_plane, count, dir_count):
        super(MultiImageNeRF, self).__init__()
        self.image_plane = image_plane
        self.render_network = RenderNetworkFW(count, dir_count)

        self.input_ch_views = dir_count

    def parameters(self):
        return self.render_network.parameters()

    def set_image_plane(self, ip):
        self.image_plane = ip

    def forward(self, x):
        input_pts, input_views = torch.split(x, [3, self.input_ch_views], dim=-1)
        x = self.image_plane(input_pts)
        return self.render_network(x, input_views)
