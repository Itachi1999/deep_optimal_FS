import torchvision.transforms as transforms
import torch

# Hyperparameters

IMG_SIZE = 224
BATCH_SIZE = 32
LR = 3e-04
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 50
NUM_CLASSES = 2
print(f"Selected Device: {DEVICE}")

# Dataset

AUGMENTATION = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
TRANSFORM = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

ROOT_DATA_DIR = 'Datasets/FNAC_Total_Dataset_'

# Computation Related
PRECISION = '32-true'
ACCELERATOR = 'gpu'
DEVICES = [0]

# Log Related
MODEL_CKPT_DIR = 'ckpt/full/'
MODEL_CKPT_FILENAME = 'model-{epoch:02d}-{val_loss:.2f}'
LOG_DIR = 'log_dir/'
LOG_NAME = 'Resnet18_Pretrained_Full'
VERSION = '0_1(32-true)'
