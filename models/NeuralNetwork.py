import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 1024),
            nn.ReLU(),
            nn.Linear(1024,  2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

"""
This are used to create synthetic datasets for testing purposes.
"""
class SyntheticData(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def generate_synthetic_dataloader(mu_1, mu_2, std_1, std_2, test_size=1000, t_size=10000):
    dataset_1 = torch.normal(mean=mu_1, std=std_1, size=(t_size, 2), dtype=torch.float32)
    dataset_1_labels = torch.full(size=(t_size,1), fill_value=0, dtype=torch.float32)
    dataset_2 = torch.normal(mean=mu_2, std=std_2, size=(t_size, 2), dtype=torch.float32)
    dataset_2_labels = torch.full(size=(t_size,1), fill_value=1, dtype=torch.float32)
    full_data = torch.cat((dataset_1, dataset_2))
    full_label = torch.cat((dataset_1_labels, dataset_2_labels))
    full_zipped = list(zip(full_data, full_label))
    np.random.shuffle(full_zipped) # Shuffle before split
    full_data = torch.stack([x[0] for x in full_zipped])
    full_label = torch.stack([x[1] for x in full_zipped])
    dataset_full = SyntheticData(full_data[:-test_size, :], full_label[:-test_size])
    dataset_validation = SyntheticData(full_data[-test_size:,:], full_label[-test_size:])
    print("Full data : {} | Validation data : {}".format(len(dataset_full), len(dataset_validation)))
    train_ = DataLoader(dataset_full, batch_size=32, shuffle=True, drop_last=True)
    valid_ = DataLoader(dataset_validation, batch_size=32, shuffle=True, drop_last=True)
    return train_, valid_