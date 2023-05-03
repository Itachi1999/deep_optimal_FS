import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch
# import numpy as np
# import config

class FNAC_DataModule(pl.LightningDataModule):
    def __init__(self, transform, augmentation, root_dir, batch_size = 32) -> None:
        super().__init__()

        self.bs = batch_size
        self.root_dir = root_dir
        self.transform = transform
        self.augmentation = augmentation

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        # return super().setup(stage)
        self.train_ds = ImageFolder(self.root_dir + '/' + 'train/', transform = self.augmentation)
        self.val_ds = ImageFolder(self.root_dir + '/' + 'validation/', transform = self.transform)
        self.test_ds = ImageFolder(self.root_dir + '/' + 'test/', transform = self.transform)


    def train_dataloader(self):
        # return super().train_dataloader()
        return DataLoader(self.train_ds, batch_size=self.bs, shuffle=True)
    
    def val_dataloader(self):
        # return super().val_dataloader()
        return DataLoader(self.val_ds, batch_size=self.bs, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.bs, shuffle=False)
    

# ds = ImageFolder('../' +config.ROOT_DATA_DIR + '/train/', transform=config.TRANSFORM)
# dl = DataLoader(ds, 1, True)

# for data, label in dl:
#     print(data, label)

class FNAC_DataModuleFull(pl.LightningDataModule):
    def __init__(self, transform, augmentation, root_dir, batch_size = 32) -> None:
        super().__init__()

        self.bs = batch_size
        self.root_dir = root_dir
        self.transform = transform
        self.augmentation = augmentation

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        # return super().setup(stage)
        generator = torch.Generator().manual_seed(42)
        entire_ds = ImageFolder(self.root_dir, transform = self.augmentation)
        self.train_ds, self.val_ds, self.test_ds = random_split(entire_ds, [112, 37, 37], generator=generator)
        


    def train_dataloader(self):
        # return super().train_dataloader()
        return DataLoader(self.train_ds, batch_size=self.bs, shuffle=True)
    
    def val_dataloader(self):
        # return super().val_dataloader()
        return DataLoader(self.val_ds, batch_size=self.bs, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.bs, shuffle=False)
    
