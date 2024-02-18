import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from spine_dataset import SpineDataset
from unet import UNet
from tqdm import tqdm
import torch_xla
import torch_xla.core.xla_model as xm
import pandas as pd
import os


def prepere_dataset(data_path, batch_size):
    train_dataset = SpineDataset(data_path)
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        train_dataset, [0.8, 0.2], generator=generator)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader


def prepare_model(device):
    return UNet(in_channels=1, num_classes=1).to(device)


def train_func(model, optimizer, criterion):
    model.train()
    batch_train_loss = []
    for img_mask in tqdm(train_dataloader):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)
        y_pred = model(img)
        optimizer.zero_grad()
        loss = criterion(y_pred, mask)
        # train_running_loss += loss.item()
        batch_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        xm.mark_step()
    return sum(batch_train_loss)/len(batch_train_loss), batch_train_loss


def validation_func(model, criterion):
    model.eval()
    batch_val_loss = []
    with torch.no_grad():
        for img_mask in tqdm(val_dataloader):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            y_pred = model(img)
            loss = criterion(y_pred, mask)
            # val_running_loss += loss.item()
            batch_val_loss.append(loss.item())
    return sum(batch_val_loss)/len(batch_val_loss), batch_val_loss


if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 128
    EPOCHS = 10
    DATA_PATH = "../SpinePatchesDataset1"
    # Update path
    CHECKPOINT_PATH = "./models/Single_SpineSegmentationv6_checkpoint_{epoch}.pth"
    MODEL_SAVE_PATH = "./models/Single_SpineSegmentationv6.pth"  # Update path
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = xm.xla_device()
    train_dataloader, val_dataloader = prepere_dataset(
        data_path=DATA_PATH, batch_size=BATCH_SIZE)
    model = prepare_model(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Load from checkpoint if it exists
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH.format(epoch=start_epoch)):
        checkpoint = torch.load(CHECKPOINT_PATH.format(epoch=start_epoch))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from checkpoint epoch {start_epoch}...")
    batch_train_losses = []
    batch_val_losses = []
    avg_train_losses = []
    avg_val_losses = []
    for epoch in tqdm(range(start_epoch, EPOCHS)):
        train_loss, batch_train_loss = train_func(
            model=model, optimizer=optimizer, criterion=criterion)
        val_loss, batch_val_loss = validation_func(
            model=model, criterion=criterion)
        batch_train_losses.extend(batch_train_loss)
        batch_val_losses.extend(batch_val_loss)
        avg_train_losses.append(train_loss)
        avg_val_losses.append(val_loss)
        print("-" * 30)
        print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print("-" * 30)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH.format(epoch=epoch))

    try:
        loss_data = {
        'AvgTrainLoss': avg_train_losses,
        'AvgValLoss': avg_val_losses,
        'BatchTrainLosses': batch_train_losses,
        'BatchValLosses': batch_val_losses
    }

        loss_df = pd.DataFrame(loss_data)
        loss_df.to_csv('./logs/loss_log.csv', index=False)
    except Exception as e:
        print(e)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
