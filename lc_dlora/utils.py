import os, zlib, torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

def merge_base_lora(alpha : torch.Tensor, beta : torch.Tensor, 
                    base : torch.Tensor, scaling):
   """
   Based on:        W_t = W_k + sAB
   """
   return torch.add(base, scaling * torch.matmul(alpha, beta))

def flatten_weight_tensor(tensor_list):
    flattened = np.concatenate([tensor.flatten().numpy() for tensor in tensor_list])
    return flattened

def check_and_create_dir(fp):
    if not os.path.exists(fp):
        os.makedirs(fp)

def compress_gzip(vector):
    return zlib.compress(vector)

def decompress_gzip(compressed_vector) -> np.array:
    decoded_vector = zlib.decompress(compressed_vector)
    return np.frombuffer(decoded_vector, dtype=np.float32) # must ensure float32 @ save.

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

