import os
import cv2 as cv
import nibabel as nib
import numpy as np
import concurrent.futures
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, recall_score, precision_score, confusion_matrix
from tqdm import tqdm
import logging

import torch

import torchvision.transforms as transforms
# from PIL import Image


class Evaluate:
    def __init__(self, data_dir, true_masks_dir,model,device):
        self.images = []
        self.true_masks = []
        self.pred_masks = []
        self.data_dir = data_dir
        self.true_masks_dir = true_masks_dir
        self.device=device
        self.model=model
        

    def normalize_volume(self, volume):
        normalized_volume = ((volume - volume.min()) /
                             (volume.max() - volume.min()) * 255).astype(np.uint8)
        return normalized_volume

    # def convert_from_3d_to_2d(self, image, is_label):
    #     processed_images = []
    #     image = self.normalize_volume(image)
    #     for z in range(image.shape[2]):
    #         slice_data = image[:, :, z]
    #         if not is_label:
    #             slice_data = cv.GaussianBlur(slice_data, (3, 3), 0)
    #         processed_images.append(slice_data)
    #     return processed_images
    def convert_from_3d_to_2d(self, volume, is_label):
        """
        Convert a 3D volume to a list of 2D slices.

        Args:
        - volume: Input 3D volume.
        - is_label: True if the volume represents labels, False otherwise.

        Returns:
        - processed_slices: List of 2D slices.
        """
        normalized_volume = self.normalize_volume(volume)
        processed_slices = []
        for z in range(normalized_volume.shape[2]):
            slice_data = normalized_volume[:, :, z]
            if not is_label:
                threshold_value = 63
                slice_data = cv.GaussianBlur(slice_data, (3, 3), 0)
                slice_data = np.where(slice_data > threshold_value, slice_data, 0)
            else:
                slice_data[slice_data>0]=255
            processed_slices.append(slice_data)
        return processed_slices
    def extract_patches(self, image, patch_size, stride):
        """
        Extract patches from a 2D image.
        Parameters:
        - image: 2D numpy array representing the image.
        - patch_size: Tuple (patch_height, patch_width) specifying the size of each patch.
        - stride: Tuple (vertical_stride, horizontal_stride) specifying the stride for patch extraction.
        Returns:
        - patches: List of extracted patches.
        """
        patches = []
        height, width = image.shape
        for i in range(0, height - patch_size[0] + 1, stride[0]):
            for j in range(0, width - patch_size[1] + 1, stride[1]):
                patch = image[i:i + patch_size[0], j:j + patch_size[1]]
                patches.append(patch)
        return patches

    def load_images_and_masks(self):
        images_paths = os.listdir(self.data_dir)
        masks_paths = os.listdir(self.true_masks_dir)
        for image_path, mask_path in zip(images_paths, masks_paths):
            image_volume = nib.load(os.path.join(
                self.data_dir, image_path)).get_fdata()
            mask_volume = nib.load(os.path.join(
                self.true_masks_dir, mask_path)).get_fdata()
            image_slices = self.convert_from_3d_to_2d(
                image_volume, is_label=False)
            mask_slices = self.convert_from_3d_to_2d(
                mask_volume, is_label=True)
            for image_slice, mask_slice in zip(image_slices, mask_slices):
                images_patches = self.extract_patches(
                    image_slice, (128, 128), (128, 128))
                mask_patches = self.extract_patches(
                    mask_slice, (128, 128), (128, 128))
                self.images.extend(images_patches)
                self.true_masks.extend(mask_patches)
        print(f" total image patches: {len(self.images)}")
        print(f" total true_masks patches: {len(self.true_masks)}")
        


    def single_image_inference(self,image, model, device):
   
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        img = transform(image).float().to(device)
        img = img.unsqueeze(0)

        with torch.no_grad():
            pred_mask = model(img)

        pred_mask = pred_mask.squeeze(0).cpu().permute(1, 2, 0)
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 0] = 1

        return pred_mask.numpy()
    
    def predict_masks(self):

        print("predicting masks................")
        for image in tqdm(self.images):
            # image_np = np.array(image)
            pred_mask = self.single_image_inference(image, self.model, self.device)
            self.pred_masks.append(pred_mask)
    
    
    def calculate_metrics(self, true_mask, pred_mask):
        # Confusion Matrix
        cm = confusion_matrix(true_mask, pred_mask, labels=[0, 1])
        # True Negative Rate (Specificity)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        # Dice coefficient
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0.0
        # Sensitivity (Recall)
        sensitivity = recall_score(true_mask, pred_mask, labels=[0, 1], average="binary", zero_division=1)
        # Precision
        precision = precision_score(true_mask, pred_mask, labels=[0, 1], average="binary", zero_division=1)
        # Accuracy
        acc_value = accuracy_score(true_mask, pred_mask)
        # F1 Score
        f1_value = f1_score(true_mask, pred_mask, labels=[0, 1], average="binary")
        # Jaccard Score
        jacc_value = jaccard_score(true_mask, pred_mask, labels=[0, 1], average="binary")

        return specificity, dice, sensitivity, precision, acc_value, f1_value, jacc_value
    
    def evaluate_metrics(self):
        print("calculating matrices ...............")
        specificity_scores = []
        dice_scores = []
        sensitivity_scores = []
        precision_scores = []
        f1_scores = []
        jacc_scores = []
        accuracy_scores = []

        for true_mask, pred_mask in tqdm(zip(self.true_masks, self.pred_masks)):
            pred_mask = pred_mask / 255.0
            pred_mask = (pred_mask > 0.5).astype(np.int32)
            pred_mask = pred_mask.flatten()

            true_mask = true_mask / 255.0
            true_mask = (true_mask > 0.5).astype(np.int32)
            true_mask = true_mask.flatten()

            specificity, dice, sensitivity, precision, acc_value, f1_value, jacc_value = self.calculate_metrics(
                true_mask, pred_mask)

            sensitivity_scores.append(sensitivity)
            dice_scores.append(dice)
            specificity_scores.append(specificity)
            precision_scores.append(precision)
            accuracy_scores.append(acc_value)
            f1_scores.append(f1_value)
            jacc_scores.append(jacc_value)

        # Calculate mean scores
        mean_sensitivity = np.mean(sensitivity_scores)
        mean_dice = np.mean(dice_scores)
        mean_specificity = np.mean(specificity_scores)
        mean_precision = np.mean(precision_scores)
        mean_accuracy = np.mean(accuracy_scores)
        mean_f1 = np.mean(f1_scores)
        mean_jacc = np.mean(jacc_scores)

        # Log scores
        logging.basicConfig(filename='score.log', level=logging.INFO)
        logging.info(f'Mean Sensitivity: {mean_sensitivity}')
        logging.info(f'Mean Dice: {mean_dice}')
        logging.info(f'Mean Specificity: {mean_specificity}')
        logging.info(f'Mean Precision: {mean_precision}')
        logging.info(f'Mean Accuracy: {mean_accuracy}')
        logging.info(f'Mean F1: {mean_f1}')
        logging.info(f'Mean Jaccard: {mean_jacc}')

        # Optionally, you can also log the individual arrays
        logging.info(f'Sensitivity Scores: {sensitivity_scores}')
        logging.info(f'Dice Scores: {dice_scores}')
        logging.info(f'Specificity Scores: {specificity_scores}')
        logging.info(f'Precision Scores: {precision_scores}')
        logging.info(f'Accuracy Scores: {accuracy_scores}')
        logging.info(f'F1 Scores: {f1_scores}')
        logging.info(f'Jaccard Scores: {jacc_scores}')

            