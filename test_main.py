from evaluation import Evaluate
import time
if __name__ == "__main__":
    data_dir = r'G:\python\3d images\niiTestData\data'
    true_masks_dir = r'G:\python\3d images\niiTestData\label'
    pred_masks_dir = r'G:\python\3d images\niiTestData\pred_masks'
    start=time.time()
    evaluate = Evaluate(
        data_dir=data_dir, true_masks_dir=true_masks_dir, pred_masks_dir=pred_masks_dir)
    evaluate.load_images_and_masks()
    
    end=time.time()

    print(f"Completed in time: {end - start} seconds")
    print(f"Start time: {start}")
    print(f"End time: {end}")