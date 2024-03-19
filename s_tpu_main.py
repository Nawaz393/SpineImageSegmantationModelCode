import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from spine_dataset import SpineDataset  
from unet import UNet  
from tqdm import tqdm
import torch_xla
import torch_xla.core.xla_model as xm
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
from sklearn.metrics import jaccard_score
import glob

def prepare_dataset(data_path, batch_size):
    """
    Prepare train and validation datasets.

    Args:
    - data_path (str): Path to the dataset.
    - batch_size (int): Batch size for DataLoader.

    Returns:
    - train_dataloader (DataLoader): DataLoader for training dataset.
    - val_dataloader (DataLoader): DataLoader for validation dataset.
    """
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
    """
    Prepare UNet model.

    Args:
    - device (torch.device): Device for model (TPU or GPU).

    Returns:
    - model (UNet): UNet model.
    """
    return UNet(in_channels=1, num_classes=1).to(device)

def dice_score(y_true, y_pred, eps=1e-7):
    """
    Compute Dice score.

    Args:
    - y_true (np.ndarray): Ground truth mask.
    - y_pred (np.ndarray): Predicted mask.
    - eps (float): Epsilon value to prevent division by zero.

    Returns:
    - dice_score (float): Dice score.
    """
    intersection = (y_true * y_pred).sum()
    return (2. * intersection + eps) / (y_true.sum() + y_pred.sum() + eps)

def load_latest_checkpoint(model, optimizer, path):
    """
    Load the latest checkpoint.

    Args:
    - model (torch.nn.Module): Model to load checkpoint weights into.
    - optimizer (torch.optim.Optimizer): Optimizer to load checkpoint state into.
    - path (str): Path to checkpoint files.

    Returns:
    - model (torch.nn.Module): Model with loaded checkpoint weights.
    - optimizer (torch.optim.Optimizer): Optimizer with loaded checkpoint state.
    - epoch (int): Epoch from which training should continue.
    """
    list_of_files = glob.glob(path) 
    if not list_of_files:
        return model, optimizer, 0
    latest_file = max(list_of_files, key=os.path.getctime)
    checkpoint = torch.load(latest_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {epoch}")
    return model, optimizer, epoch

def train_func(model, optimizer, criterion, writer, epoch):
    """
    Train the model for one epoch.

    Args:
    - model (torch.nn.Module): Model to train.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - criterion (torch.nn.Module): Loss function.
    - writer (SummaryWriter): TensorBoard writer for logging.
    - epoch (int): Current epoch.

    Returns:
    - train_loss (float): Average training loss.
    - batch_train_loss (list): List of training losses for each batch.
    """
    model.train()
    batch_train_loss = []
    for img_mask in tqdm(train_dataloader):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)
        y_pred = model(img)
        optimizer.zero_grad()
        loss = criterion(y_pred, mask)
        batch_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        xm.mark_step()
    for i, loss in enumerate(batch_train_loss):
        writer.add_scalar('BatchLoss/Train', loss, epoch * len(train_dataloader) + i)
    return sum(batch_train_loss) / len(batch_train_loss), batch_train_loss

def validation_func(model, criterion, writer, epoch):
    """
    Validate the model on validation dataset.

    Args:
    - model (torch.nn.Module): Model to validate.
    - criterion (torch.nn.Module): Loss function.
    - writer (SummaryWriter): TensorBoard writer for logging.
    - epoch (int): Current epoch.

    Returns:
    - val_loss (float): Average validation loss.
    - batch_val_loss (list): List of validation losses for each batch.
    - val_dice (float): Average Dice score.
    - batch_val_dice (list): List of Dice scores for each batch.
    - val_jaccard (float): Average Jaccard score.
    - batch_val_jaccard (list): List of Jaccard scores for each batch.
    """
    model.eval()
    batch_val_loss = []
    batch_val_dice = []
    batch_val_jaccard = []
    with torch.no_grad():
        for img_mask in tqdm(val_dataloader):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            y_pred = model(img)
            loss = criterion(y_pred, mask)
            batch_val_loss.append(loss.item())

            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).float()
            dice = dice_score(mask.cpu().numpy(), y_pred.cpu().numpy())
            jaccard = jaccard_score(y_pred.cpu().numpy().flatten(), mask.cpu().numpy().flatten())
            batch_val_dice.append(dice)
            batch_val_jaccard.append(jaccard)
    for i, loss in enumerate(batch_val_loss):
        writer.add_scalar('BatchLoss/Validation', loss, epoch * len(val_dataloader) + i)
    for i, dice in enumerate(batch_val_dice):
        writer.add_scalar('BatchDice/Validation', dice, epoch * len(val_dataloader) + i)
    for i, jaccard in enumerate(batch_val_jaccard):
        writer.add_scalar('BatchJaccard/Validation', jaccard, epoch * len(val_dataloader) + i)

    return (
        sum(batch_val_loss) / len(batch_val_loss),
        batch_val_loss,
        sum(batch_val_dice) / len(batch_val_dice),
        batch_val_dice,
        sum(batch_val_jaccard) / len(batch_val_jaccard),
        batch_val_jaccard
    )

if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 128
    EPOCHS = 15
    DATA_PATH = "../SpinePatchesDataset1"
    # Update path
    CHECKPOINT_PATH = "./models/Single_SpineSegmentationv6_checkpoint_{epoch}.pth"
    MODEL_SAVE_PATH = "./models/Single_SpineSegmentationv8.pth"  # Update path

    device = xm.xla_device()
    train_dataloader, val_dataloader = prepare_dataset(
        data_path=DATA_PATH, batch_size=BATCH_SIZE)
    model = prepare_model(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    writer = SummaryWriter(log_dir='./logs/tensorboard_logs')

    # Load from checkpoint if it exists
    l_CHECKPOINT_PATH = "./models/Single_SpineSegmentationv6_checkpoint_*.pth"
    model, optimizer, start_epoch = load_latest_checkpoint(model, optimizer, l_CHECKPOINT_PATH)

    batch_train_losses = []
    batch_val_losses = []
    batch_val_dices = []
    batch_val_jaccard = []
    avg_train_losses = []
    avg_val_losses = []
    avg_val_dices = []
    avg_val_jaccard = []
    for epoch in tqdm(range(start_epoch, EPOCHS)):
        train_loss, batch_train_loss = train_func(
            model=model, optimizer=optimizer, criterion=criterion, writer=writer, epoch=epoch)
        val_loss, batch_val_loss, val_dice, batch_val_dice, val_jaccard, batch_val_jaccard = validation_func(
            model=model, criterion=criterion, writer=writer, epoch=epoch)
        batch_train_losses.extend(batch_train_loss)
        batch_val_losses.extend(batch_val_loss)
        batch_val_dices.extend(batch_val_dice)
        batch_val_jaccard.extend(batch_val_jaccard)
        avg_train_losses.append(train_loss)
        print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print(f"Valid Dice EPOCH {epoch + 1}: {val_dice:.4f}")
        print(f"Valid Jaccard EPOCH {epoch + 1}: {val_jaccard:.4f}")
        print("-" * 30)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH.format(epoch=epoch))
    writer.close()
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
