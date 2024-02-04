from lc_dlora.Config import Config
from lc_dlora.lora.LoraManager import LoraManager
from lc_dlora.lc.DeltaManager import DeltaManager
from torch.utils.data.dataloader import DataLoader
from lc_dlora.CheckpointManager import CheckpointManager
from lc_dlora.metrics.metrics import accuracy, accuracy_binary
import torch

class ParameterNotFoundException(Exception):
    def __init__(self, param):
        super().__init__("Parameter: {} cannot be found".format(param))

class Trainer:
    def __init__(self, config : Config):
        self.config = config
        self.evaluation_log = []

    def get_optimizer(self, model_params):
        match self.config.optimizer:
            case "sgd":
                return torch.optim.SGD(model_params, lr = self.config.learning_rate)
            case "adam":
                return torch.optim.Adam(model_params, lr = self.config.learning_rate)
            case "rmsprop":
                return torch.optim.RMSprop(model_params, lr = self.config.learning_rate)
            case _:
                raise ParameterNotFoundException(self.config.optimizer)

    def get_evaluation_function(self):
        match self.config.evaluation_function:
            case "accuracy":
                return accuracy
            case "binary_accuracy":
                return accuracy_binary
            case _:
                raise ParameterNotFoundException(self.config.evaluation_function)
            
    def get_loss(self):
        match self.config.loss_function:
            case "binary_cross_entropy":
                return torch.nn.BCELoss()
            case "cross_entropy":
                return torch.nn.CrossEntropyLoss()
            case _:
                raise ParameterNotFoundException(self.config.loss_function)
        
    def get_evaluation_log(self):
        return self.evaluation_log

    def train(self, model, train_loader : DataLoader,
            validation_loader : DataLoader = None):
        epochs = self.config.epochs
        set_id, node_id = 0, 0
        lora_manager = LoraManager(model, self.config)
        checkpoint_manager = CheckpointManager(self.config)
        checkpoint_manager.save_base(model)
        lora_model = lora_manager.createLoraModelFromBase()
        initial_bases = lora_manager.extractLoraDeltaBases(lora_model)
        delta_manager = DeltaManager(self.config, initial_bases)  
        # Initial superstep save.
        checkpoint_manager.save_super_step(sd=lora_model.state_dict(), 
                                                       set_id=set_id, iteration=0, epoch=0)
        optimizer = self.get_optimizer(model.parameters())
        loss_function = self.get_loss()
        evaluation_function = self.get_evaluation_function()
        
        # Expected save pattern with superset = 5 ([s] == superset):
        # [s], 0, 1, 2, 3, 4, [s], 0, 1, 2, 3, 4, [s], 0, 1, 2, 3, 4
        
        for epoch in range(epochs):
            for iter, data in enumerate(train_loader):
                print("Epoch {} | Iteration {}".format(epoch, iter))
                inputs, labels = data
                optimizer.zero_grad()
                outputs = lora_model(inputs)
                if self.config.loss_function == "binary_cross_entropy":
                    outputs = torch.sigmoid(outputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                if node_id == self.config.super_step: # Superstep
                    set_id, node_id = set_id + 1, 0
                    checkpoint_manager.save_super_step(sd=lora_model.state_dict(), 
                                                       set_id=set_id, iteration=iter, epoch=epoch)
                    model = lora_manager.mergeLoraModel(lora_model)
                    optimizer = self.get_optimizer(lora_model.parameters()) # Reset optimizer

                else: # Normal checkpoint process
                    node_id += 1
                    current_bases = lora_manager.extractLoraDeltaBases(lora_model)
                    promoted_delta_full, \
                        promoted_delta_decomposed = delta_manager.take_delta(current_bases)
                    bias = lora_manager.extractBias(lora_model)
                    checkpoint_manager.save_delta(delta=(promoted_delta_full, 
                                                promoted_delta_decomposed), bias=bias,
                                                node_id=node_id, set_id=set_id,
                                                iteration=iter,epoch=epoch)
                if self.config.in_training_validation:
                    if iter % self.config.validation_frequency == 0:
                        print("Running validation for {} Epoch, {} Iteration...".format(epoch, iter))
                        res = evaluation_function(lora_model, validation_loader)
                        print("Results: {}".format(res))
                        self.evaluation_log.append(res)
        print("Final evaluation: {}".format(evaluation_function(lora_model, validation_loader)))
        checkpoint_manager.log_training_log()
        
