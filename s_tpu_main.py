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

if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 32
    EPOCHS = 5
    DATA_PATH = "../SpinePatchesDataset1"
    CHECKPOINT_PATH = "./models/Single_SpineSegmentationv5_checkpoint_{epoch}.pth"  # Update path
    MODEL_SAVE_PATH = "./models/Single_SpineSegmentationv5.pth"  # Update path
    LOG_FILE_PATH = "./logs/loss_log.xlsx"  # Update path
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = xm.xla_device()

    train_dataset = SpineDataset(DATA_PATH)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet(in_channels=1, num_classes=1).to(device)
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

    # Initialize a DataFrame to store losses
    loss_df = pd.DataFrame(columns=['Epoch', 'Batch', 'Train Loss', 'Valid Loss'])

    for epoch in tqdm(range(start_epoch, EPOCHS)):
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
            xm.mark_step()

            # Save batch loss to DataFrame
            loss_df = loss_df.append({'Epoch': epoch + 1, 'Batch': idx + 1, 'Train Loss': loss.item()}, ignore_index=True)

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

        # Save epoch loss to DataFrame
        loss_df = loss_df.append({'Epoch': epoch + 1, 'Batch': 'Epoch', 'Train Loss': train_loss, 'Valid Loss': val_loss}, ignore_index=True)

        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH.format(epoch=epoch))

    # Save the DataFrame to an Excel file
    loss_df.to_excel(LOG_FILE_PATH, index=False)

    # Final save to the desired path
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
