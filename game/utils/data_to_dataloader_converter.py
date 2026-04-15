import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.save_load_data import load_tensor_dataset

class RoomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __getshape__(self):
        return self.X.shape, self.y.shape


def get_dataloader(path, batch_size=32):
    X, y = load_tensor_dataset(path)

    dataset = RoomDataset(X, y)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    return dataloader