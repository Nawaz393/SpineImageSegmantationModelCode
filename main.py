import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from spine_dataset import SpineDataset
from unet import UNet
from tqdm import tqdm


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


def train(model, optimizer, criterion, train_dataloader, val_dataloader, device, epochs):
    """
    Train the model.

    Args:
    - model (torch.nn.Module): Model to train.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - criterion (torch.nn.Module): Loss function.
    - train_dataloader (DataLoader): DataLoader for training dataset.
    - val_dataloader (DataLoader): DataLoader for validation dataset.
    - device (str): Device to use for training ('cuda' or 'cpu').
    - epochs (int): Number of epochs to train.

    Returns:
    - model (torch.nn.Module): Trained model.
    """
    for epoch in range(epochs):
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            y_pred = model(img)
            optimizer.zero_grad()
            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                y_pred = model(img)
                loss = criterion(y_pred, mask)
                val_running_loss += loss.item()
            val_loss = val_running_loss / (idx + 1)

        print("-" * 30)
        print(f"Train Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print("-" * 30)

    return model


def main():
    # Parameters
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 32
    EPOCHS = 1
    DATA_PATH = "./dataset"
    MODEL_SAVE_PATH = "./models/unet1.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare datasets
    train_dataloader, val_dataloader = prepare_dataset(
        data_path=DATA_PATH, batch_size=BATCH_SIZE)

    # Initialize model, optimizer, and criterion
    model = UNet(in_channels=1, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Train the model
    trained_model = train(model, optimizer, criterion,
                          train_dataloader, val_dataloader, device, epochs=EPOCHS)

    # Save the trained model
    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
