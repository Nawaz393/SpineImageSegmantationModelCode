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
from torch.nn.parallel import DistributedDataParallel as DDP


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    batch_train_losses = []

    for img_mask in tqdm(train_loader):
        img, mask = img_mask[0].float().to(device), img_mask[1].float().to(device)

        optimizer.zero_grad()
        y_pred = model(img)
        loss = criterion(y_pred, mask)
        running_loss += loss.item()

        batch_train_losses.append(loss.item())

        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)

    avg_loss = running_loss / len(train_loader)

    return avg_loss, batch_train_losses


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0
    batch_val_losses = []

    with torch.no_grad():
        for img_mask in tqdm(val_loader):
            img, mask = img_mask[0].float().to(device), img_mask[1].float().to(device)

            y_pred = model(img)
            loss = criterion(y_pred, mask)
            running_loss += loss.item()

            batch_val_losses.append(loss.item())

    avg_loss = running_loss / len(val_loader)

    return avg_loss, batch_val_losses


def plot_epoch_losses(train_losses, val_losses, epoch):
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
    MODEL_SAVE_PATH = "./models/SpineSegmentationv3.pth"

    device = xm.xla_device()
    train_dataset = SpineDataset(DATA_PATH)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    
    # parallel_loader = pl.ParallelLoader(train_loader, [device])
    # train_loader = parallel_loader.per_device_loader(device)
    # parallel_loader_val = pl.ParallelLoader(val_loader, [device])
    # val_loader = parallel_loader_val.per_device_loader(device)

    model = UNet(in_channels=1, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(EPOCHS)):
        print(f"epoch {epoch}")
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        # Create new ParallelLoader for each epoch
        parallel_loader = pl.ParallelLoader(train_loader, [device])
        train_loader = parallel_loader.per_device_loader(device)
        parallel_loader_val = pl.ParallelLoader(val_loader, [device])
        val_loader = parallel_loader_val.per_device_loader(device)
        train_loss, batch_train_losses = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,  batch_val_losses = validate_epoch(model, val_loader, criterion, device)

        train_losses.extend(batch_train_losses)
        val_losses.extend(batch_val_losses)

        if xm.is_master_ordinal():
            print("-" * 30)
            print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f}")
            print(f"Valid Loss EPOCH {epoch + 1}: {val_loss:.4f}")
            print("-" * 30)

    save_losses_to_excel(train_losses, 'all_batch_losses.xlsx')
    avg_train_losses = [sum(train_losses) / len(train_losses)] * len(train_losses)
    avg_val_losses = [sum(val_losses) / len(val_losses)] * len(val_losses)
    save_losses_to_excel(avg_train_losses, 'avg_train_losses.xlsx')
    save_losses_to_excel(avg_val_losses, 'avg_val_losses.xlsx')

    xm.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == "__main__":
    print("Training Started...")
    xmp.spawn(
        main,
        args=(),
        start_method='fork',
    )
    print("Training Completed.")
