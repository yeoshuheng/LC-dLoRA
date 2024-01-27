from lc_dlora.Config import Config
from lc_dlora.CheckpointManager import CheckpointManager
import os, json

class Restore:

    def __init__(self, config : Config):
        self.config = config
        self.checkpoint_manager = CheckpointManager(config)
        self.training_log = self.load_existing_training_log()
    
    def load_existing_training_log(self):
        with open(os.path.join(self.maindir, "log.json"), "r") as f:
            training_log = json.load(f)
        return training_log
    
    def view_logs(self):
        iter_epoch = self.training_log.keys()
        output_dict = {}
        for epoch, iteration in iter_epoch:
            if epoch not in output_dict:
                output_dict[epoch] = []
            output_dict[epoch].append(iteration)
        output = '\n'.join("Epoch {} : \n {}".format(k, 
                        ', '.join(output_dict[k])) for k in output_dict.keys())
        print(output)

    def restore(self, iteration, epoch, fresh_model):
        node_id, set_id = self.training_log[(epoch, iteration)]
        if node_id == -1:
            self.checkpoint_manager.restore_checkpoint_superstep(set_id, fresh_model)
        else:
            self.checkpoint_manager.restore_checkpoint(node_id, set_id, fresh_model)


