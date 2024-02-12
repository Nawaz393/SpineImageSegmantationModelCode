import torch
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from unet import UNet


#%%


def single_image_inference(image_pth, model_pth, mask_pth, device):
    model = UNet(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    img = transform(Image.open(image_pth)).float().to(device)
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred_mask = torch.sigmoid(model(img))
        pred_mask = (pred_mask > 0.01).float()

    img = img.squeeze(0).cpu().permute(1, 2, 0)
    pred_mask = pred_mask.squeeze(0).cpu().permute(1, 2, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input Image")

    axes[1].imshow(pred_mask, cmap="gray")
    axes[1].set_title("Predicted Mask")

    axes[2].imshow(Image.open(mask_pth), cmap="gray")
    axes[2].set_title("Ground Truth Mask")

    plt.show()


if __name__ == "__main__":
    SINGLE_IMG_PATH = "./dataset/processed_data/data_patch_38282.png"
    MODEL_PATH = "./models/unet1.pth"
    MASK_PATH = "./dataset/processed_label/label_patch_38282.png"
    SAVE_FOLDER = "./saved_images/"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, MASK_PATH, device)

#%%
