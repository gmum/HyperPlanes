import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=256, planes=0):
        super().__init__()
        self.num_layers = 4
        self.is_training = True
        self.planes = planes
        # input layer
        self.add_module('{0}_{1}'.format(0,0), nn.Conv2d(x_dim, hid_dim, 3, padding=1))   
        self.add_module('{0}_{1}'.format(0,1), nn.BatchNorm2d(hid_dim))
        # hidden layer
        for i in [1, 2]:
            self.add_module('{0}_{1}'.format(i,0), nn.Conv2d(hid_dim, hid_dim, 3, padding=1))   
            self.add_module('{0}_{1}'.format(i,1), nn.BatchNorm2d(hid_dim))     
        # last layer
        self.add_module('{0}_{1}'.format(3,0), nn.Conv2d(hid_dim, z_dim, 3, padding=1))   
        self.add_module('{0}_{1}'.format(3,1), nn.BatchNorm2d(z_dim))     
        if planes !=0:
            self.conv_transpose = nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=17, stride=17, padding=4)
        self.o3 = nn.Conv2d(3, 3*32, 5, 1)
    
    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        if not isinstance(params, OrderedDict):
            params = OrderedDict(params)

        output = x
        for i in range(self.num_layers):
            weight_key = '{0}_{1}.weight'.format(i, 0)
            bias_key = '{0}_{1}.bias'.format(i, 0)

            conv_weight = params[weight_key]
            conv_bias = params[bias_key]

            output = F.conv2d(output, conv_weight, bias=conv_bias, padding=1)

            weight_key = '{0}_{1}.weight'.format(i, 1)
            bias_key = '{0}_{1}.bias'.format(i, 1)

            bn_weight = params[weight_key]
            bn_bias = params[bias_key]

            output = F.batch_norm(output, weight=bn_weight, bias=bn_bias,
                                running_mean=self._modules['{0}_{1}'.format(i, 1)].running_mean,
                                running_var=self._modules['{0}_{1}'.format(i, 1)].running_var,
                                training=self.is_training)
            output = F.relu(output)
            output = F.max_pool2d(output, 2)

        third_output = self.o3(x)

        first_output = F.avg_pool2d(output, 8)  # AveragePool Here
        if self.planes!=0:
            second_output = self.conv_transpose(output)
        # Additional convolution layers to reshape output
        #output = output.view(x.size(0), -1)
        #output = output.reshape(64, -1)
        first_output = first_output.view(-1, x.size(0))
        if self.planes!=0:
            second_output = second_output.permute(0, 3, 2, 1)
            second_output = second_output[:self.planes]
        else:
            second_output=None
        return first_output, second_output, third_output





