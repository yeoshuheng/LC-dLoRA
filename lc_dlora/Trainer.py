from lc_dlora.Config import Config
from lc_dlora.lora.LoraManager import LoraManager
from lc_dlora.lc.DeltaManager import DeltaManager
from torch.utils.data.dataloader import DataLoader
from lc_dlora.CheckpointManager import CheckpointManager

class Trainer:
    def __init__(self, config : Config):
        self.config = config

    def train(self, model, optimizer, train_loader : DataLoader, 
              validation_loader : DataLoader, loss_function : function,
              evaluation_function : function):
        epochs = self.config.epochs
        set_id, node_id = 0, 0
        lora_manager = LoraManager(model, self.config)
        checkpoint_manager = CheckpointManager(self.config)
        checkpoint_manager.save_base(model)
        lora_model = lora_manager.createLoraModelFromBase()
        initial_bases = lora_manager.extractLoraDeltaBases(lora_model)
        delta_manager = DeltaManager(self.config, initial_bases)  

        # Super step save for first set (LoRA weights + conv only)
        # Expected save pattern with superset = 5:
        # [s], 0, 1, 2, 3, 4, [s], 0, 1, 2, 3, 4, [s], 0, 1, 2, 3, 4
        
        for epoch in range(epochs):
            for iter, data in enumerate(train_loader):
                print("Epoch {} | Iteration {}".format(epoch, iter))
                inputs, labels = data
                optimizer.zero_grad()
                outputs = lora_model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                # Superstep
                if node_id == self.config.super_step:
                    set_id, node_id = set_id + 1, 0

                    # Super step save for subsequent sets.
                    
                else:
                    # Normal checkpoint process
                    current_bases = lora_manager.extractLoraDeltaBases(lora_model)
                    promoted_delta_full, \
                        promoted_delta_decomposed = delta_manager.take_delta(current_bases)
                    bias = lora_manager.extractBias(lora_model)
                    checkpoint_manager.save_delta(delta=(promoted_delta_full, 
                                                   promoted_delta_decomposed), bias=bias,
                                                   node_id=node_id, set_id=set_id)
                    node_id += 1

        # End of training => retrieve training logs from checkpoint manager.

        
