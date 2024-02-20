import os
import cv2 as cv
import nibabel as nib
import numpy as np
import concurrent.futures


class Evaluate:
    def __init__(self, data_dir, true_masks_dir, pred_masks_dir):
        self.images = []
        self.true_masks = []
        self.pred_masks = []
        self.data_dir = data_dir
        self.true_masks_dir = true_masks_dir
        self.pred_masks_dir = pred_masks_dir
        

    def normalize_volume(self, volume):
        normalized_volume = ((volume - volume.min()) /
                             (volume.max() - volume.min()) * 255).astype(np.uint8)
        return normalized_volume

    def convert_from_3d_to_2d(self, image, is_label):
        processed_images = []
        image = self.normalize_volume(image)
        for z in range(image.shape[2]):
            slice_data = image[:, :, z]
            if not is_label:
                slice_data = cv.GaussianBlur(slice_data, (3, 3), 0)
            processed_images.append(slice_data)
        return processed_images

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
        # Iterate through the image with the specified stride and extract patches
        for i in range(0, height - patch_size[0] + 1, stride[0]):
            for j in range(0, width - patch_size[1] + 1, stride[1]):
                patch = image[i:i + patch_size[0], j:j + patch_size[1]]
                patches.append(patch)
        return patches

    def load_images_and_masks(self):
        images_paths = os.listdir(self.data_dir)
        masks_paths = os.listdir(self.true_masks_dir)

        def process_image_and_mask(image_path, mask_path):
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

                for image_patch, mask_patch in zip(images_patches, mask_patches):
                    self.images.append(image_patch)
                    self.true_masks.append(mask_patch)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(process_image_and_mask, images_paths, masks_paths)
        print(f" total mask patches: {len(self.true_masks)}")
        print(f" total image patches: {len(self.images)}")
