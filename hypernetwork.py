import torch
import torch.nn as nn
from collections import OrderedDict

class HN(nn.Module):
    def __init__(self, depth, width, hdim: int, target_layers, embedding_size):
        super().__init__()
        self.head_len = depth
        self.hidden_size= width
        self.hdim = hdim
        self.target_layers = target_layers
        self.embedding_size = embedding_size

        if self.head_len == 1:
            layers = [nn.Linear(self.embedding_size, self.hdim + 1)]

        else:
            layers = [
                nn.Linear(self.embedding_size, self.hidden_size),
                nn.ReLU()
            ]
            for i in range(self.head_len - 2):
                layers.extend([nn.Linear(self.hidden_size,  self.hidden_size), nn.ReLU()])
            layers.append(nn.Linear(self.hidden_size, self.hdim + 1)) #  * self.embedding_size

        self.hn = nn.Sequential(*layers)

    def forward(self, support_embeddings: torch.Tensor):
        out = self.hn(support_embeddings)
        return out
    



def calculate_hypernetwork_output(layer: list, target_network):
    target_layers = OrderedDict(target_network.state_dict())
    filtered_layers = {
        l: target_layers[l].shape for l in list(target_layers)[2:] if l in layer
    }  # skip first layer
    output_weights_w = sum(
        [v[1] for (l, v) in filtered_layers.items() if "weight" in l]
    )
    bias_layers = [v[0] for (l, v) in filtered_layers.items() if "bias" in l]
    output_weights_b = len(bias_layers)
    output_width = output_weights_w + output_weights_b
    return max(bias_layers), output_width, filtered_layers