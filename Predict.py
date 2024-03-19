import os
import torch
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm

from unet import UNet  

class Predict:
    def __init__(self, data_dir, pred_dir, model, device):
        """
        Initialize Predict object.

        Args:
        - data_dir (str): Directory containing input images.
        - pred_dir (str): Directory to save predicted masks.
        - model (torch.nn.Module): Pre-trained model for prediction.
        - device (torch.device): Device to perform inference on.
        """
        self.data_dir = data_dir
        self.pred_dir = pred_dir
        self.model = model
        self.device = device
        self.data_paths = os.listdir(self.data_dir)
        if not os.path.exists(self.pred_dir):
            os.mkdir(self.pred_dir)
    
    def single_image_inference(self, image, model, device):
        """
        Perform inference on a single image.

        Args:
        - image (np.ndarray): Input image.
        - model (torch.nn.Module): Pre-trained model for prediction.
        - device (torch.device): Device to perform inference on.

        Returns:
        - pred_mask (np.ndarray): Predicted mask.
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        img = transform(image).float().to(device)
        img = img.unsqueeze(0)

        with torch.no_grad():
            pred_mask = model(img)

        pred_mask = pred_mask.squeeze(0).cpu().permute(1, 2, 0).squeeze(-1)
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 0] = 1

        return pred_mask.numpy()
    
    def predict(self):
        """
        Perform prediction on all images in the data directory.
        """
        for data_path in tqdm(self.data_paths, total=len(self.data_dir)):
            image = cv2.imread(os.path.join(self.data_dir, data_path), cv2.IMREAD_GRAYSCALE)
            pred_mask = self.single_image_inference(image, self.model, self.device)
            cv2.imwrite(os.path.join(self.pred_dir, data_path), pred_mask * 255)
            print(f"Predicted {data_path}")
        print("Prediction complete")
    

if __name__ == "__main__":

    data_dir =  r'G:\python\3d images\SpineTestPatches4\data'
    pred_dir = r'G:\python\3d images\SpineTestPatches4\pred2'
    model_pth = r'G:\python\models\Single_SpineSegmentationv9_cpu.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(
        model_pth, map_location=torch.device(device)))
    model.eval()
    predictor = Predict(data_dir, pred_dir, model, device)
    predictor.predict()
