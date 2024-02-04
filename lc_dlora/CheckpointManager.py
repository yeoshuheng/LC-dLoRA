from lc_dlora.Config import Config
from lc_dlora.utils import check_and_create_dir, compress_gzip, \
    decompress_gzip, flatten_weight_tensor, merge_base_lora
import os, pickle, torch, json
import numpy as np

"""
    `main_dir` in config is relative to working directory, save is also done relative to working directory,
    hence the correct `main_dir` may be different from that used in training.
    Users should set up configuration as needed.
"""

class CheckpointManager:

    def __init__(self, config : Config):
        self.maindir = config.main_dir
        self.decomposed_layers = config.decomposed_layers
        check_and_create_dir(self.maindir)
        self.training_log = {}
        self.rank = config.rank
        self.scaling = config.scaling
    
    def log_training_log(self):
        with open(os.path.join(self.maindir, "log.json"), "w") as f:
            json.dump(self.training_log, f)
    
    def save_delta(self, delta : tuple, bias, node_id, set_id, iteration, epoch):
        compressed_delta_full = compress_gzip(delta[0])
        compressed_delta_decomposed = compress_gzip(delta[1])
        setdir = self.maindir + "/set{}".format(set_id)
        check_and_create_dir(setdir)
        checkpoint_name = "lc-dlora_checkpoint_set_{}_node_{}.pt".format(set_id, node_id)
        sp = os.path.join(setdir, checkpoint_name)
        with open(sp, "wb") as f:
            pickle.dump((compressed_delta_full, 
                         compressed_delta_decomposed, bias), f)
        self.training_log["{}-{}".format(epoch, iteration)] = "{}-{}".format(node_id, set_id)

    def save_super_step(self, sd, set_id, iteration, epoch):
        setdir = self.maindir + "/set{}".format(set_id)
        check_and_create_dir(setdir)
        checkpoint_name = "lc-dlora_snapshot_set_{}.pt".format(set_id)
        sp = os.path.join(setdir, checkpoint_name)
        torch.save(sd, sp)
        self.training_log["{}-{}".format(epoch, iteration)] \
            = "{}-{}".format(-1, set_id) # Super set saves do not have a node id.

    def load_delta(self, node_id, set_id) -> tuple:
        setdir = self.maindir + "/set{}".format(set_id)
        checkpoint_name = "lc-dlora_checkpoint_set_{}_node_{}.pt".format(set_id, node_id)
        sp = os.path.join(setdir, checkpoint_name)
        with open(sp, "rb") as f:
            compressed_delta_full, compressed_delta_decomposed, bias = pickle.load(f)
        delta_full, delta_decomposed = decompress_gzip(compressed_delta_full), \
                        decompress_gzip(compressed_delta_decomposed)
        return (delta_full, delta_decomposed, bias)

    def load_super_step(self, set_id) -> dict:
        setdir = self.maindir + "/set{}".format(set_id)
        checkpoint_name = "lc-dlora_snapshot_set_{}.pt".format(set_id)
        sp = os.path.join(setdir, checkpoint_name)
        return torch.load(sp)
    
    def save_base(self, base):
        torch.save(base.state_dict(), self.maindir + "/base.pt")
    
    def restore_checkpoint_superstep(self, set_id, model) -> dict:
        model.load_state_dict(self.load_super_step(set_id))

    def extract_alpha_beta_superstep(self, sd : dict):
        alpha_betas = []
        for name in self.decomposed_layers:
            alpha_betas.append(sd["parameter_dict." + name + ".alpha"])
            alpha_betas.append(sd["parameter_dict." + name + ".alpha"])
        return flatten_weight_tensor(alpha_betas)
    
    def extract_non_decomposed(self, sd : dict):
        full_weights = []
        for name, layer in sd.values():
            if name not in self.decomposed_layers:
                full_weights.append(layer)
        return flatten_weight_tensor(full_weights)
        
    def restore_checkpoint(self, node_id, set_id, model):
        decomposed_layers = self.decomposed_layers
        setdirs = [self.load_super_step(i) for i in range(0, set_id + 1)]
        lora_snapshots = [self.extract_alpha_beta_superstep(x) for x in setdirs]
        base_model_sd = torch.load(self.maindir + "/base.pt")
        decomposed_base = np.sum(lora_snapshots, axis = 0)
        full_base = self.extract_non_decomposed(self.load_super_step(set_id))
        checkpoints = [self.load_delta(i, set_id) for i in range(0, node_id + 1)]
        final_bias = checkpoints[-1][-1]
        for full_delta, decomposed_delta, _ in checkpoints:
            full_base = np.add(full_base, full_delta)
            decomposed_base = np.add(decomposed_base, decomposed_delta)
        new_sd = model.state_dict()
        last_idx, last_idx_dcomp = 0, 0
        for name, param in new_sd.values():
            if "bias" in name:
                new_sd[name] = final_bias[name]
                continue
            dim = param.numpy().shape
            if not dim:
                continue
            if name in decomposed_layers:
                t_element_alpha = dim[0] * self.rank
                t_element_beta = dim[1] * self.rank
                alpha = decomposed_base[last_idx_dcomp : last_idx_dcomp + t_element_alpha]
                last_idx_dcomp += t_element_alpha
                beta = decomposed_base[last_idx_dcomp : last_idx_dcomp + t_element_beta]
                last_idx_dcomp += t_element_beta
                alpha = torch.unflatten(torch.from_numpy(np.copy(alpha)), -1, (dim[0], self.rank))
                beta = torch.unflatten(torch.from_numpy(np.copy(beta)), -1, (self.rank, dim[1]))
                restored_decomp = merge_base_lora(alpha, beta, base_model_sd[name], self.scaling)
                new_sd[name] = restored_decomp
                continue
            else:
                t_elements = np.prod(dim)
                needed_ele = full_base[last_idx : last_idx + t_elements]
                new_sd[name] = torch.unflatten(torch.from_numpy(np.copy(needed_ele)), -1, dim)
                last_idx += t_elements
        model.load_state_dict(new_sd)