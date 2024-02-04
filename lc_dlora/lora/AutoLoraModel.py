import torch.nn as nn
from lc_dlora.Config import Config
from collections import deque
from lc_dlora.lora.DecomposedLinear import DecomposedLinear

class LoraConstructionException(Exception):
    def __init__(self):
        self.message = "Failure to construct layers, check your decomposed layer settings."

class AutoLoraModel(nn.Module):
    def __init__(self, model : nn.Module, config : Config):
        super().__init__()
        self.to_decompose = config.decomposed_layers
        self.lora_scaling = config.scaling
        self.lora_rank = config.rank
        self.parameter_dict = nn.ParameterDict() # NOTE state_dict keys have added 'parameter_dict.' in front.
        self.call_order = deque()
        self.decomposed_layers = []
        self.full_layers = []
        for name, param in model.named_modules():
            if isinstance(param, nn.Sequential) or name == "": continue
            if name in self.to_decompose:
                self.decomposed_layers.append(name)
                lora_layer = self.generateLoraLayer(param)
                param = lora_layer
            else:
                self.full_layers.append(name)
            self.parameter_dict[name] = param
            self.call_order.append(name)
    
    def generateLoraLayer(self, module : nn.Module) -> DecomposedLinear:
        if not hasattr(module, 'in_features') or not hasattr(module, 'out_features'):
            raise LoraConstructionException()
        in_channels, out_channels = module.in_features, module.out_features
        weight, bias = module.weight, module.bias
        return DecomposedLinear(in_shape=in_channels, out_shape=out_channels, 
                                base=weight, bias=bias,
                                rank=self.lora_rank, scaling=self.lora_scaling)

    def forward(self, x):
        for module_name in self.call_order:
            x = self.parameter_dict[module_name](x)
        return x