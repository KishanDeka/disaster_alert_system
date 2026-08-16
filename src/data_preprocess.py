import os
import shutil
import numpy as np 
import pandas as pd
import kagglehub
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class CSVDataset(Dataset):
    def __init__(self, csv_file, split='train', transform=None):
        self.csv_file = csv_file
        self.split = split
        self.transform = transform
        
        # Load CSV and filter by split
        df = pd.read_csv(csv_file)
        self.data = df[df['split'] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.loc[idx, 'filepath']
        label = int(self.data.loc[idx, 'label'])
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

    @staticmethod
    def prepare_dataset(target_dir="../data", dataset_handle="kishandeka27/aiderv2-dataset", 
                           csv_name="data_list.csv" ):
        """Downloads dataset via kagglehub and generates the index CSV."""
        os.makedirs(target_dir, exist_ok=True)
        csv_path = os.path.join(target_dir, csv_name)

        # Download dataset via kagglehub
        os.environ["KAGGLEHUB_CACHE"] = os.path.abspath(target_dir)
        downloaded_path = kagglehub.dataset_download(dataset_handle)

        # split directories and class labels
        splits = {
            "train": os.path.join(downloaded_path, "Train", "Train"),
            "val": os.path.join(downloaded_path, "Val", "Val"),
            "test": os.path.join(downloaded_path, "Test", "Test")
        }

        labels = {
            0: "Fire",
            1: "Flood",
            2: "Normal",
            3: "Earthquake",
        }

        # Index image filepaths into data frame
        data = []
        for split, split_path in splits.items():
            for label, label_name in labels.items():
                folder_path = os.path.join(split_path, label_name)
                
                if not os.path.exists(folder_path):
                    continue

                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        data.append({
                            "filename": img_name,
                            "filepath": img_path,
                            "split": split,
                            "label_name": label_name,
                            "label": label
                        })

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Dataset indexed successfully ({len(df)} samples) -> {csv_path}")
        return csv_path, splits, labels

    @classmethod
    def calculate_mean_and_std(cls, csv_file, split='train', img_size=(128, 128), batch_size=64):
        """Calculates channel-wise dataset statistics dynamically."""
        temp_transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor()
        ])

        temp_dataset = cls(csv_file=csv_file, split=split, transform=temp_transform)
        temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

        channels_sum, channels_squared_sum, num_batches = 0, 0, 0

        for images, _ in temp_loader:
            channels_sum += torch.mean(images, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(images ** 2, dim=[0, 2, 3])
            num_batches += 1

        mean = (channels_sum / num_batches).tolist()
        std = ((channels_squared_sum / num_batches - (channels_sum / num_batches) ** 2) ** 0.5).tolist()
        
        return mean, std
