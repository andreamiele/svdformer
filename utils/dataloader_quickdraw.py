import numpy as np
import torch.utils.data
import os

def read_npy(file_path):
    # Function to read npy file
    return np.load(file_path)

class QuickDrawDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
        self.transforms = transforms
        self.file_list = self._get_file_list()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = read_npy(file_path)
        if self.transforms:
            data = self.transforms(data)
        return data

    def _get_file_list(self):
        # Create a list of file paths from the data directory
        file_list = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith('.npy')]
        return file_list

class QuickDrawDataLoader(object):
    def __init__(self, data_path, batch_size, num_workers):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data_loader(self):
        dataset = QuickDrawDataset(data_path=self.data_path)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=self.batch_size,
                                                  num_workers=self.num_workers,
                                                  shuffle=True)
        return data_loader

# Usage Example
data_path = 'path/to/quickdraw/data'
quickdraw_loader = QuickDrawDataLoader(data_path, batch_size=32, num_workers=4)
train_data_loader = quickdraw_loader.get_data_loader()
