import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, jaccard_score, f1_score
from skimage.io import imread
import logging

class ScoreCalculator:
    def __init__(self, true_masks_dir, pred_masks_dir, log_file, score_file):
        """
        Initialize ScoreCalculator object.
        
        Args:
        - true_masks_dir (str): Directory containing true masks.
        - pred_masks_dir (str): Directory containing predicted masks.
        - log_file (str): Path to the log file.
        - score_file (str): Path to the file to write scores.
        """
        self.true_masks_dir = true_masks_dir
        self.pred_masks_dir = pred_masks_dir
        self.accuracy_scores = []
        self.precision_scores = []
        self.jaccard_scores = []
        self.dice_scores = []
        self.f1_scores = []
        self.sensitivity_scores = []
        self.specificity_scores = []
        self.log_file = log_file
        self.score_file = score_file
        logging.basicConfig(filename=self.log_file, level=logging.INFO)

    def calculate_scores(self):
        """
        Calculate evaluation scores.
        """
        for t_path, p_path in zip(sorted(os.listdir(self.true_masks_dir), key=lambda x: int(x.split('_')[-1].split('.')[0])), sorted(os.listdir(self.pred_masks_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))):
            true_mask = imread(os.path.join(self.true_masks_dir, t_path))
            pred_mask = imread(os.path.join(self.pred_masks_dir, p_path))

            if np.sum(true_mask) == 0:
                continue
            true_mask = true_mask.astype(np.float32) / 255
            pred_mask = pred_mask.astype(np.float32) / 255
            true_mask = (true_mask > 0).astype(int).flatten()
            pred_mask = (pred_mask > 0).astype(int).flatten()
            self.accuracy_scores.append(accuracy_score(true_mask, pred_mask))
            self.precision_scores.append(precision_score(
                true_mask, pred_mask, zero_division=1))
            self.jaccard_scores.append(jaccard_score(true_mask, pred_mask))
            self.dice_scores.append(self._dice_score(true_mask, pred_mask))
            self.f1_scores.append(f1_score(true_mask, pred_mask))
            sensitivity, specificity = self._sensitivity_specificity(
                true_mask, pred_mask)
            self.sensitivity_scores.append(sensitivity)
            self.specificity_scores.append(specificity)

    def _dice_score(self, true_mask, pred_mask):
        """
        Calculate Dice score.
        
        Args:
        - true_mask (np.ndarray): Ground truth mask.
        - pred_mask (np.ndarray): Predicted mask.
        
        Returns:
        - dice_score (float): Dice score.
        """
        intersection = np.sum(true_mask * pred_mask)
        return (2. * intersection) / (np.sum(true_mask) + np.sum(pred_mask))

    def _sensitivity_specificity(self, true_mask, pred_mask):
        """
        Calculate sensitivity and specificity.
        
        Args:
        - true_mask (np.ndarray): Ground truth mask.
        - pred_mask (np.ndarray): Predicted mask.
        
        Returns:
        - sensitivity (float): Sensitivity score.
        - specificity (float): Specificity score.
        """
        tp = np.sum((true_mask == 1) & (pred_mask == 1))
        tn = np.sum((true_mask == 0) & (pred_mask == 0))
        fp = np.sum((true_mask == 0) & (pred_mask == 1))
        fn = np.sum((true_mask == 1) & (pred_mask == 0))
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return sensitivity, specificity

    def log_scores(self):
        """
        Log scores to a file.
        """
        logging.info(f'Accuracy: {np.mean(self.accuracy_scores)}')
        logging.info(f'Precision: {np.mean(self.precision_scores)}')
        logging.info(f'Jaccard: {np.mean(self.jaccard_scores)}')
        logging.info(f'Dice: {np.mean(self.dice_scores)}')
        logging.info(f'F1: {np.mean(self.f1_scores)}')
        logging.info(f'Sensitivity: {np.mean(self.sensitivity_scores)}')
        logging.info(f'Specificity: {np.mean(self.specificity_scores)}')

    def write_scores(self):
        """
        Write scores to a file.
        """
        with open(self.score_file, 'w') as f:
            f.write(f'Accuracy: {np.mean(self.accuracy_scores)}\n')
            f.write(f'Precision: {np.mean(self.precision_scores)}\n')
            f.write(f'Jaccard: {np.mean(self.jaccard_scores)}\n')
            f.write(f'Dice: {np.mean(self.dice_scores)}\n')
            f.write(f'F1: {np.mean(self.f1_scores)}\n')
            f.write(f'Sensitivity: {np.mean(self.sensitivity_scores)}\n')
            f.write(f'Specificity: {np.mean(self.specificity_scores)}\n')

# Usage:
score_calculator = ScoreCalculator(true_masks_dir=r'G:\python\3d images\SpineTestPatches4\label',
                                   pred_masks_dir=r'G:\python\3d images\SpineTestPatches4\pred2', log_file='scores4.log', score_file="score1.txt")
score_calculator.calculate_scores()
score_calculator.log_scores()
score_calculator.write_scores()
