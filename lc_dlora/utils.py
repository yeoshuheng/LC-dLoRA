import os, zlib, torch
import numpy as np

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

