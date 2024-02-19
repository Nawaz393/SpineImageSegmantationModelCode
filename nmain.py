import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torch import optim, nn
import os
from spine_dataset import SpineDataset
from unet import UNet
import numpy as np


def save_batch_losses_to_excel(all_batch_losses, file_path):
    df = pd.DataFrame({'Batch Losses': all_batch_losses})
    df.to_excel(file_path, index=False, header=False,
                mode='a')  # Append to the existing file


def save_epoch_loss_to_excel(epoch_loss, file_path):
    df = pd.DataFrame({'Epoch Loss': [epoch_loss]})
    df.to_excel(file_path, index=False, header=False, mode='a')


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0
    batch_train_losses = []

    for idx, img_mask in enumerate(train_loader):
        img, mask = img_mask[0].float().to(
            device), img_mask[1].float().to(device)

        optimizer.zero_grad()

        y_pred = model(img)
        loss = criterion(y_pred, mask)
        running_loss += loss.item()

        batch_train_losses.append(loss.item())

        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)

    avg_loss = running_loss / len(train_loader)

    # Gather losses from all processes and compute the mean
    all_batch_train_losses = xm.mesh_reduce(
        'train_losses', batch_train_losses, np.mean)
    avg_loss = xm.mesh_reduce('avg_train_loss', avg_loss, np.mean)

    # Save all batch losses to Excel file for this epoch
    save_batch_losses_to_excel(all_batch_train_losses, 'batch_losses.xlsx')

    return avg_loss, all_batch_train_losses


def validate_epoch(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0
    batch_val_losses = []

    with torch.no_grad():
        for idx, img_mask in enumerate(val_loader):
            img, mask = img_mask[0].float().to(
                device), img_mask[1].float().to(device)

            y_pred = model(img)
            loss = criterion(y_pred, mask)
            running_loss += loss.item()

            batch_val_losses.append(loss.item())

    avg_loss = running_loss / len(val_loader)

    # Gather losses from all processes and compute the mean
    all_batch_val_losses = xm.mesh_reduce(
        'val_losses', batch_val_losses, np.mean)
    avg_loss = xm.mesh_reduce('avg_val_loss', avg_loss, np.mean)

    # Save all batch losses to Excel file for this epoch
    save_batch_losses_to_excel(all_batch_val_losses, 'batch_losses.xlsx')

    return avg_loss, all_batch_val_losses

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss - Batches')
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - Epoch {epoch}')

    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss - Batches')
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.title(f'Validation Loss - Epoch {epoch}')

    plt.tight_layout()
    plt.show()


def save_losses_to_excel(losses, file_path):
    df = pd.DataFrame({'Loss': losses})
    df.to_excel(file_path, index=False)


def main(index):
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 64
    EPOCHS = 10
    DATA_PATH = "../SpinePatchesDataset1"
    MODEL_SAVE_PATH = "./models/SpineSegmentationv1.pth"

    device = xm.xla_device()
    train_dataset = SpineDataset(DATA_PATH)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        train_dataset, [0.8, 0.2], generator=generator)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              )
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=BATCH_SIZE,
                            )
    model = UNet(in_channels=1, num_classes=1).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = pl.MpDeviceLoader(
        train_loader,
        device)
    val_loader = pl.MpDeviceLoader(
        train_loader,
        device)
    for epoch in tqdm(range(EPOCHS)):
        train_loss, all_batch_train_losses = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch)
        val_loss,  all_batch_val_losses = validate_epoch(
            model, val_loader, criterion, device, epoch)

        if xm.is_master_ordinal():
            # Print information only once from the master process
            print("-" * 30)
            print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f}")
            print(f"Valid Loss EPOCH {epoch + 1}: {val_loss:.4f}")
            print("-" * 30)

            # Save average losses across epochs to Excel file
            save_epoch_loss_to_excel(train_loss, 'epoch_losses.xlsx')
            save_epoch_loss_to_excel(val_loss, 'epoch_losses.xlsx')

    # Save additional metrics or model state if needed
    # For example, you can save accuracy, model parameters, etc.
    if xm.is_master_ordinal():
        xm.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == "__main__":
    # os.environ['PJRT_DEVICE'] = 'TPU'
    print("training Started.....")
    xmp.spawn(
        main,
        args=(),
        nprocs=8,
        start_method='fork',


    )
    print("training completed.....")
