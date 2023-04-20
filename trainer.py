import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import Codes.config as config
from Codes.model import FeatureExtractor
from Codes.dataModule import FNAC_DataModule

logger = TensorBoardLogger(
    save_dir=config.LOG_DIR, name=config.LOG_NAME, version=config.VERSION
)

callbacks = [EarlyStopping(monitor='train_loss'), ModelCheckpoint(dirpath=config.MODEL_CKPT_DIR, filename=config.MODEL_CKPT_FILENAME, monitor='val_loss', mode="min", save_top_k = 3)]

dm = FNAC_DataModule(config.TRANSFORM, config.AUGMENTATION, root_dir=config.ROOT_DATA_DIR, batch_size=config.BATCH_SIZE)

num_classes = config.NUM_CLASSES
model = FeatureExtractor(num_classes=num_classes, lr=config.LR, img_size=224)

trainer = pl.Trainer(accelerator=config.ACCELERATOR, devices=config.DEVICES, precision=config.PRECISION, callbacks=callbacks, logger=logger, min_epochs=1, max_epochs=config.NUM_EPOCHS)


trainer.fit(model=model, datamodule=dm)
trainer.validate(model=model, datamodule=dm)
trainer.test(model=model, datamodule=dm)

