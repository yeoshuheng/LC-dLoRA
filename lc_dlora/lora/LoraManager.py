from lc_dlora.Config import Config
from lc_dlora.lora import AutoLoraModel
import numpy as np
import copy, torch
from lc_dlora.utils import flatten_weight_tensor, merge_base_lora

class LoraManager:

    def __init__(self, model, config : Config):
        self.decomposed_layers = config.decomposed_layers
        self.config = config
        self.lora_rank = config.rank
        self.lora_scaling = config.scaling
        self.base_model = model  # We keep the model to convert here.
        self.original_state_dictionary = copy.deepcopy(model.state_dict())

    def createLoraModelFromBase(self) -> AutoLoraModel:
        """
        Creates a low-rank version of the original model.
        """
        return AutoLoraModel(self.base_model, self.config)
    
    def extractLoraDeltaBases(self, model : AutoLoraModel):
        decomposed = model.decomposed_layers
        full = model.full_layers
        module_map = model.parameter_dict
        decomposed_tensor_list = []
        fullstate_tensor_list = []
        for layer in decomposed:
            decomposed_linear_module = module_map[layer]
            decomposed_tensor_list.append(decomposed_linear_module.alpha.detach())
            decomposed_tensor_list.append(decomposed_linear_module.beta.detach())
        for layer in full:
            if layer not in module_map:
                continue # Skip pass pooling & ReLU layers.
            full_module = module_map[layer]
            if not hasattr(full_module, 'weight'):
                continue
            fullstate_tensor_list.append(full_module.weight.detach().clone())
        return (flatten_weight_tensor(fullstate_tensor_list), \
            flatten_weight_tensor(decomposed_tensor_list))
    
    def extractBias(self, model : torch.nn.Module):
        ret = {}
        for name, param in model.state_dict().items():
            if "bias" in name:
                ret[name] = param
        return ret

    def mergeLoraModel(self, active_model : AutoLoraModel) -> AutoLoraModel:
        """
        Creation of the new branch point takes advantage of availability of 
        in-training current active model. Removes the need of conducting a full restore.
        """
        active_sd = copy.deepcopy(active_model.state_dict())
        rebuild_dict = copy.deepcopy(self.original_state_dictionary)
        need_to_restore = self.decomposed_layers
        for name, weight in rebuild_dict.items():
            if "bias" in name:
                rebuild_dict[name] = active_sd["parameter_dict." + name]
                continue
            temp = ".".join(name.split(".")[:-1]) # Gets rid of weight / bias in front.
            if temp in need_to_restore:
                base = weight
                alpha = active_sd["parameter_dict." + temp + ".alpha"]
                beta = active_sd["parameter_dict." + temp + ".beta"]
                rebuild_dict[name] = merge_base_lora(
                    alpha=alpha , beta=beta,
                    base=base, scaling=self.lora_scaling)
            else: 
                rebuild_dict[name] = active_sd["parameter_dict." + name]
        self.original_state_dictionary = rebuild_dict
        self.base_model.load_state_dict(rebuild_dict)
        return self.createLoraModelFromBase()