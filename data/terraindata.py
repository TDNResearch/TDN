import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math


def array_train_test_split(X_data, y_data, test_size = 0.3):
    index = range(len(X_data))
    test_index = random.choices(index, k=math.floor(len(X_data)*test_size))
    train_index = [i for i in index if i not in test_index ]
    X_test = np.array([X_data[i] for i in test_index])
    y_test = np.array([y_data[i] for i in test_index])
    X_train = np.array([X_data[i] for i in train_index])
    y_train = np.array([y_data[i] for i in train_index])
    return X_train, X_test, y_train, y_test


class TerrainDataset(Dataset):

    
    def __init__(self, input_imgs, label_imgs):

        self.input_imgs = input_imgs
        self.label_imgs = label_imgs

    def __len__(self):
        return len(self.input_imgs)
    
    

    def __getitem__(self, idx):
        input_img, label_img = self.input_imgs[idx], self.label_imgs[idx]
        return {
            "input":input_img, 
            "output":label_img}

class TerrainDataModule(pl.LightningDataModule):
    
    def __init__(self, input_imgs, label_imgs , batch_size=2):

        super().__init__()

        self.input_imgs = input_imgs
        self.label_imgs = label_imgs
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.X_train, X_test, self.y_train, y_test = array_train_test_split(self.input_imgs, self.label_imgs, test_size = 0.2)
        self.X_val, self.X_test, self.y_val, self.y_test = array_train_test_split(X_test, y_test, test_size = 0.5)
        

    def train_dataloader(self):
        return DataLoader(
            dataset = TerrainDataset(self.X_train, self.y_train), 
            shuffle = True, 
            num_workers = 2,
            batch_size = self.batch_size
            )

    def val_dataloader(self):
        return DataLoader(
            dataset = TerrainDataset(self.X_val, self.y_val), 
            shuffle = False, 
            num_workers = 2,
            batch_size = self.batch_size
            )

    def test_dataloader(self):
        return DataLoader(
            dataset = TerrainDataset(self.X_test, self.y_test), 
            shuffle = False, 
            num_workers = 2,
            batch_size = self.batch_size
            )