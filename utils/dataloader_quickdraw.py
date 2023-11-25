
import os
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms

class QuickDrawDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        # Load data from all .npy files in the folder
        all_data = []
        for file in os.listdir(self.folder_path):
            if file.endswith('.npy'):
                file_path = os.path.join(self.folder_path, file)
                data = np.load(file_path, encoding='latin1', allow_pickle=True)
                all_data.extend(data)
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        drawing = sample['drawing']
        processed_drawing = self.process_drawing(drawing)

        if self.transform:
            processed_drawing = self.transform(processed_drawing)

        return processed_drawing

    def process_drawing(self, drawing):
        # Implement the conversion of drawing data here
        processed_drawing = np.array(drawing)  # Placeholder
        return processed_drawing
