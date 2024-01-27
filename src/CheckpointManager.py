from src.Config import Config
from src.utils import check_and_create_dir, compress_gzip, decompress_gzip
import os, pickle, torch

class CheckpointManager:

    def __init__(self, config : Config):
        self.maindir = config.main_dir
        check_and_create_dir(self.maindir)
    
    def save_delta(self, delta : tuple, bias, node_id, set_id):
        compressed_delta_full = compress_gzip(delta[0])
        compressed_delta_decomposed = compress_gzip(delta[1])
        setdir = self.maindir + "/set{}".format(set_id)
        check_and_create_dir(setdir)
        checkpoint_name = "lc-dlora_checkpoint_set_{}_node_{}.pt".format(set_id, node_id)
        sp = os.path.join(setdir, checkpoint_name)
        with open(sp, "wb") as f:
            pickle.dump((compressed_delta_full, 
                         compressed_delta_decomposed, bias), f)

    def save_super_step(self, sd, set_id):
        setdir = self.maindir + "/set{}".format(set_id)
        check_and_create_dir(setdir)
        checkpoint_name = "lc-dlora_snapshot_set_{}.pt".format(set_id)
        sp = os.path.join(setdir, checkpoint_name)
        torch.save(sd, sp)

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

    def restore_checkpoint(self, node_id, set_id):
        pass