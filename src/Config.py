"""
This class represents a LC-dLoRA Trainer configuration.
The following parameters are needed.

    epochs : The number of epochs the model is being trained for.
    scaling : The scaling factor of dLoRA layers. (Influence dLoRA weights has on the model)
    branch_path : The file path to the model at the branching point.
    main_dir : The directory to store all checkpoint sets.
    decomposed_layers : A list representing the layers we need to decompose into LoRA.
    device : Device to train the model on.
    
    "epochs" :  ,
    "scaling" : 0.5,
    "branch_path" : ,
    "branch_dir" : ,
    "decomposed_layers" : ,
    "rank" : 8,
    "device" : "cpu", 
    "learning_rate" : 0.01,
"""

class Config:
    
    def __init__(self, config_dict : dict):
        self.epochs = config_dict["epochs"]
        self.scaling = config_dict["scaling"]
        self.branch_path = config_dict["branch_path"]
        self.main_dir = config_dict["branch_dir"]
        self.decomposed_layers = config_dict["decomposed_layers"]
        self.rank = config_dict["rank"]
        self.device = config_dict["device"]
        self.learning_rate = config_dict["learning_rate"]