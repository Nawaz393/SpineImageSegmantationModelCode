from evaluation import Evaluate
import time
# import torch_xla
# import torch_xla.core.xla_model as xm
from unet import UNet
import torch
if __name__ == "__main__":
    # data_dir = r'../niiTestData/data'
    # true_masks_dir = '../niiTestData/label'
    model_pth = "../models/Single_SpineSegmentationv6.pth"
    data_dir=r'G:\python\3d images\niiTestData\data'
    true_masks_dir=r'G:\python\3d images\niiTestData\label'
    model_pth = "../models/Single_SpineSegmentationv7.pth"
    # device = xm.xla_device()
    device="cpu"
    
    start = time.time()
    
    model = UNet(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(
        model_pth, map_location=torch.device(device)))
    model.eval()
   
    
    evaluate = Evaluate(
        data_dir=data_dir, true_masks_dir=true_masks_dir, model=model, device=device)
    
    evaluate.load_images_and_masks()
    
    evaluate.predict_masks()
    evaluate.save_images("./testPatches")
    evaluate.evaluate_metrics()
  
    
    end = time.time()
    
    print(f"Completed in time: {end - start} seconds")
    print(f"Start time: {start}")
    print(f"End time: {end}")