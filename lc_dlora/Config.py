"""
This class represents a LC-dLoRA Trainer configuration.
The following parameters are needed.

    epochs : The number of epochs the model is being trained for.
    scaling : The scaling factor of dLoRA layers. (Influence dLoRA weights has on the model)
    main_dir : The directory to store all checkpoint sets.
    decomposed_layers : A list representing the layers we need to decompose into LoRA.
    device : Device to train the model on.
    learning_rate : Optimizer learning rate.
    lc_bw : The bitwidth for lc-checkpoint mechanism.
    super_step : The number of checkpoints between each super step save.
    training_log : Where logs are stored, can be used in conjunction with 
        checkpoint manager to restore model at a certain iteration / epoch.
    in_training_validation : If validation set is being used in training.
    validation_frequency : How often we need to run the validation step. (eg. if validation_frequency = n,
        we run the validation function every n iterations)
    loss_function : Which loss function to use, pick from : ["accuracy"]
    optimizer : Which optimizer to use, pick from : ["adam", "rmsprop", "sgd"]

    ===== Template =====

    config_dict = {
        "epochs" :  3,
        "scaling" : 0.5,
        "main_dir" : HHD + "/dlora/checkpoints/",
        "decomposed_layers" :  TEST_DECOMPOSED,
        "rank" : 8,
        "device" : "cpu", 
        "learning_rate" : 0.01,
        "lc_bw" : 3,
        "super_step" : 10,
        "training_log" : HHD + "/dlora/logs/",
        "base_path" : HHD + "/dlora/checkpoints/",
        "in_training_validation" : True,
        "validation_frequency" : 100,
        "loss_function" : "accuracy",
        "optimizer" : "sgd"
    } 
"""

class Config:
    
    def __init__(self, config_dict : dict):
        self.epochs = config_dict["epochs"]
        self.scaling = config_dict["scaling"]
        self.main_dir = config_dict["main_dir"]
        self.decomposed_layers = config_dict["decomposed_layers"]
        self.rank = config_dict["rank"]
        self.device = config_dict["device"]
        self.learning_rate = config_dict["learning_rate"]
        self.lc_bw = config_dict["lc_bw"]
        self.super_step = config_dict["super_step"]
        self.training_log_dir = config_dict["training_log"]
        self.loss_function = config_dict["loss_function"]
        self.optimizer = config_dict["optimizer"]
        self.in_training_validation = config_dict["in_training_validation"]
        self.validation_frequency = config_dict["validation_frequency"]
        self.evaluation_function = config_dict["evaluation_function"]

"""
Experiment configurations
"""
TEST_DECOMPOSED_NN = ['linear_relu_stack.2', 'linear_relu_stack.4', 'linear_relu_stack.6',
                      'linear_relu_stack.8', 'linear_relu_stack.10']

SYNTHETIC_DATASET_CONFIG = Config({
        "epochs" :  1,
        "scaling" : 0.5,
        "main_dir" : "/volumes/Ultra Touch/dlora/checkpoints/",
        "decomposed_layers" : TEST_DECOMPOSED_NN,
        "rank" : 8,
        "device" : "cpu", 
        "learning_rate" : 0.01,
        "lc_bw" : 3,
        "super_step" : 10,
        "training_log" : "/volumes/Ultra Touch/dlora/logs/",
        "base_path" : "/volumes/Ultra Touch/dlora/checkpoints/",
        "in_training_validation" : True,
        "validation_frequency" : 3,
        "loss_function" : "binary_cross_entropy",
        "optimizer" : "sgd",
        "evaluation_function" : "binary_accuracy"
    })