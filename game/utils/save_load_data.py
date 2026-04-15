import json
import os
import numpy as np

def load_json_dataset(path):
    with open(path, "r") as f:
        dataset = json.load(f)

    print(f"Loaded {path} with {len(dataset)} samples")
    return dataset

def save_json_dataset(dataset, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(dataset, f)

    print(f"\nJSON Dataset saved to {path}")

def save_tensor_dataset(tensors, labels, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    np.savez(path, X=tensors, y=labels)
    print(f"Saved tensor dataset to {path}")

def load_tensor_dataset(path):
    data = np.load(path)
    tensors = data['X']
    labels = data['y']
    print(f"Loaded {path} with {len(tensors)} samples")
    return tensors, labels