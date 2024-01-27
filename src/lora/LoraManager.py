from src.Config import Config
from src.lora import AutoLoraModel
import numpy as np
import copy, torch

def flatten_weight_tensor(tensor_list):
    flattened = np.concatenate([tensor.flatten().numpy() for tensor in tensor_list])
    return flattened

def merge_base_lora(alpha : torch.Tensor, beta : torch.Tensor, 
                    base : torch.Tensor, scaling):
   """
   Based on:        W_t = W_k + sAB
   """
   return torch.add(base, scaling * torch.matmul(alpha, beta))

class LoraManager:

    def __init__(self, model, config : Config):
        self.decomposed_layers = config.decomposed_layers
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
            decomposed_tensor_list.append(decomposed_linear_module.alpha)
            decomposed_tensor_list.append(decomposed_linear_module.beta)
        for layer in full:
            if layer not in module_map:
                continue # Skip pass pooling & ReLU layers.
            full_module = module_map[layer]
            if not hasattr(full_module, 'weight'):
                continue
            fullstate_tensor_list.append(full_module.weight)
        return flatten_weight_tensor(fullstate_tensor_list), \
            flatten_weight_tensor(decomposed_tensor_list)

    def mergeLoraModel(self, active_model : AutoLoraModel) -> AutoLoraModel:
        """
        Creation of the new branch point takes advantage of availability of 
        in-training current active model. Removes the need of conducting a full restore.
        """
        active_sd = copy.deepcopy(active_model.state_dict())
        rebuild_dict = copy.deepcopy(self.original_state_dictionary)
        need_to_restore = self.decomposed_layers
        for name, weight in rebuild_dict.values():
            if name in need_to_restore:
                base = weight
                alpha = active_sd["parameter_dict." + name + ".alpha"]
                beta = active_sd["parameter_dict." + name + ".alpha"]
                rebuild_dict[name] = merge_base_lora(
                    alpha=alpha , beta=beta,
                    base=base, scaling=self.lora_scaling)
            else: 
                rebuild_dict[name] = active_sd["parameter_dict." + name]
        self.original_state_dictionary = rebuild_dict
        self.model.load_state_dict(rebuild_dict)
        return self.createLoraModelFromBase()