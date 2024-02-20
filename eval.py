import torch
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from unet import UNet
import torch_xla
import torch_xla.core.xla_model as xm

import os


def single_image_inference(image_pth, model_pth, mask_pth, device):
    model = UNet(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(
        model_pth, map_location=torch.device(device)))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    img = transform(Image.open(image_pth)).float().to(device)
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred_mask = model(img)

    img = img.squeeze(0).cpu().permute(1, 2, 0)
    pred_mask = pred_mask.squeeze(0).cpu().permute(1, 2, 0)
    pred_mask[pred_mask < 0] = 0
    pred_mask[pred_mask > 0] = 1

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input Image")

    axes[1].imshow(pred_mask, cmap="gray")
    axes[1].set_title("Predicted Mask")

    axes[2].imshow(Image.open(mask_pth), cmap="gray")
    axes[2].set_title("Ground Truth Mask")

    plt.show(block=True)
    return pred_mask.numpy()


if __name__ == "__main__":
    num = 101215
    SINGLE_IMG_PATH = f"../SpinePatchesDataset1/data/data_patch_{num}.png"
    MODEL_PATH = "./models/Single_SpineSegmentationv6.pth"
    MASK_PATH = f"../SpinePatchesDataset1/label/label_patch_{num}.png"
    device = xm.xla_device()

    pred_mask = single_image_inference(
        SINGLE_IMG_PATH, MODEL_PATH, MASK_PATH, device)
