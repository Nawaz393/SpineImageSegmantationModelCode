import os
from unet import UNet
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from spine_dataset import SpineDataset
import torch_xla
import torch_xla.core.xla_model as xm

def save_model():
    CHECKPOINT_PATH = "./models/Single_SpineSegmentationv6_checkpoint_{9}.pth"
    MODEL_SAVE_PATH = "./models/Single_SpineSegmentationv7.pth"
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 128
    EPOCHS = 11
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = xm.xla_device()
    
    model = UNet(in_channels=1, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    if os.path.exists(CHECKPOINT_PATH.format(epoch=9)):
        checkpoint = torch.load(CHECKPOINT_PATH.format(epoch=9))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    model.to("cpu")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

