import pytorch_lightning as pl
import torchvision.models as models
import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
import torchmetrics
import numpy as np
import os
from tqdm import tqdm


class FeatureExtractor(pl.LightningModule):
    def __init__(self, num_classes, lr, img_size) -> None:
        super().__init__()

        # self.feature_extractor = torch.nn.Sequential(*(list(models.resnet18(weights='ResNet18_Weights.DEFAULT', progress = True).children())[:-1]))
        self.feature_extractor = models.resnet18(pretrained = True, progress = True)
        self.feature_extractor.fc = nn.Identity()
        self.classifier_layer = nn.Linear(512, num_classes)
        self.num_classes = num_classes
        self.lr = lr
        self.save_hyperparameters()
        self.acc = torchmetrics.Accuracy('multiclass', num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.softmax(self.classifier_layer(x), dim=1)

        return x
        # print(self.feature_extractor)
        # print(x.shape)

    def _common_step(self, batch):
        imgs, labels = batch
        scores = self.forward(imgs)
        loss = self.criterion(scores, labels)

        return loss, scores, labels
        

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss, scores, labels = self._common_step(batch)
        acc = self.acc(scores, labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True)
        
        if batch_idx % 5 == 0:
            imgs = x[:25]
            grid = torchvision.utils.make_grid(imgs)
            self.logger.experiment.add_image('FNAC_training_Images', grid, self.global_step)

        return loss

    
    def validation_step(self, batch, batch_idx):
        loss, scores, labels = self._common_step(batch)
        acc = self.acc(scores, labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, scores, labels = self._common_step(batch)
        acc = self.acc(scores, labels)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        self.log('test_acc', acc, prog_bar=True, on_epoch=True)
       
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def feature_extraction(self, dataLoader, stage:str, filePath:str):
        features = []
        labels_list = []
        print(dataLoader.batch_size)
        assert dataLoader.batch_size == 1

        if not os.path.exists(filePath):
            os.mkdir(filePath)

        for i, batch in enumerate(tqdm(dataLoader), 0):
            with torch.no_grad():

                inputs, labels = batch
                output = self.feature_extractor(inputs)
                output = torch.flatten(output).detach().numpy()
                features.append(output)
                labels = torch.flatten(labels).detach().numpy()
                labels_list.append(labels)
        features = np.stack(features, axis=0)
        labels_list = np.stack(labels_list, axis=0)
        print(features.shape)
        print(labels_list.shape)
        np.save(f'{filePath}/{stage}_resnet18_fnacFull_features.npy', features)
        np.save(f'{filePath}/{stage}_resnet18_fnacFull_labels.npy', labels_list)


    


# data = torch.randn(16, 3, 224, 224)
# model = FeatureExtractor(2, 11, 224)
# model(data)

