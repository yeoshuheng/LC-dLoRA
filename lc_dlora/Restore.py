from lc_dlora.Config import Config
from lc_dlora.CheckpointManager import CheckpointManager
import os, json

class Restore:

    def __init__(self, config : Config):
        self.config = config
        self.checkpoint_manager = CheckpointManager(config)
        self.training_log = self.load_existing_training_log()
    
    def load_existing_training_log(self):
        with open(os.path.join(self.config.main_dir, "log.json"), "r") as f:
            training_log = json.load(f)
        return training_log
    
    def view_logs(self):
        iter_epoch = self.training_log.keys()
        output_dict = {}
        for epoch_iteration in iter_epoch:
            epoch, iteration = epoch_iteration.split("~")
            if epoch not in output_dict:
                output_dict[epoch] = []
            output_dict[epoch].append(iteration)
        output = '\n'.join("Epoch {} : \n {}".format(k, 
                        ', '.join(output_dict[k])) for k in output_dict.keys())

    def restore(self, iteration, epoch, model):
        #print(self.training_log["{}~{}".format(epoch, iteration)])
        node_id, set_id = self.training_log["{}~{}".format(epoch, iteration)].split("~")
        node_id, set_id = int(node_id), int(set_id)
        self.checkpoint_manager.restore_checkpoint(node_id, set_id, model)


