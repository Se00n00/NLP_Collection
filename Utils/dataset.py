import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset


class Amazon_Dataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset  # Dataset object

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]  # Access row as dict
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'labels': torch.tensor(item['label'])
        }
