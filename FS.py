import pytorch_lightning as pl
from Codes.model import FeatureExtractor
from Codes.dataModule import FNAC_DataModuleFull
import Codes.config as config


best_model = FeatureExtractor.load_from_checkpoint(checkpoint_path = 'ckpt/full/model-epoch=06-val_loss=0.35.ckpt')

dm = FNAC_DataModuleFull(transform=config.TRANSFORM, augmentation=config.AUGMENTATION, root_dir=config.ROOT_DATA_DIR, batch_size=1)

# print(dm.setup())
# print(dm.train_dataloader())
# print(dm.val_dataloader())

# best_model.feature_extraction()
dm.setup()
best_model.freeze()
best_model.feature_extraction(dm.train_dataloader(), stage="train", filePath='features/full')
best_model.feature_extraction(dm.val_dataloader(), stage='val', filePath='features/full')
best_model.feature_extraction(dm.test_dataloader(), stage='test', filePath='features/full')
