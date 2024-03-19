import os
import cv2 as cv
import numpy as np
import nibabel as nib
import time
from tqdm import tqdm

class PreProcessing:
    def __init__(self):
        self.count = 0
        
    def normalize_volume(self, volume):
        """
        Normalize the volume to the range [0, 1].
        Args:
        - volume: Input 3D volume.
        Returns:
        - normalized_volume: Normalized 3D volume.
        """
        # Normalize volume to range [0, 255] for displaying as an image
        volume = ((volume - volume.min()) / (volume.max() - volume.min()) * 255).astype(np.uint8)
        return volume

    def convert_from_3d_to_2d(self, image, is_label):
        """
        Convert 3D volume slices to 2D images.
        Args:
        - image: Input 3D volume.
        - is_label: Boolean indicating if the image represents a label.
        Returns:
        - processed_images: List of processed 2D images.
        """
        processed_images = []
        image = self.normalize_volume(image)  # Normalize the input image
        for z in range(image.shape[2]):  # Iterate through the z-axis of the image
            slice_data = image[:, :, z]  # Extract a 2D slice from the 3D volume
            if not is_label:
                # Apply preprocessing steps for non-label images
                slice_data = np.where(slice_data > 63, slice_data, 0)  # Thresholding
                slice_data = cv.GaussianBlur(slice_data, (3, 3), 0)  # Gaussian blur
            processed_images.append(slice_data)  # Append processed slice to the list
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
                patch = image[i:i + patch_size[0], j:j + patch_size[1]]  # Extract patch from image
                patches.append(patch)  # Append patch to the list
        return patches

    def convert_and_save_patches(self, input_data_dir, input_label_dir, output_data_dir, output_label_dir,
                                 patch_size=(128, 128),
                                 stride=(32, 32)):
        """
        Convert 3D volumes to 2D images, extract patches, and save them.
        Args:
        - input_data_dir: Directory containing input data volumes.
        - input_label_dir: Directory containing input label volumes.
        - output_data_dir: Directory to save data patches.
        - output_label_dir: Directory to save label patches.
        - patch_size: Tuple (patch_height, patch_width) specifying the size of each patch.
        - stride: Tuple (vertical_stride, horizontal_stride) specifying the stride for patch extraction.
        """
        data_files = os.listdir(input_data_dir)
        label_files = os.listdir(input_label_dir)

        print(data_files, label_files)
        os.makedirs(output_data_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        # Iterate through each pair of data and label files
        for data_file, label_file in tqdm(zip(data_files, label_files)):
            data_volume = nib.load(os.path.join(input_data_dir, data_file)).get_fdata()
            label_volume = nib.load(os.path.join(input_label_dir, label_file)).get_fdata()
            data_slices = self.convert_from_3d_to_2d(data_volume, False)  # Convert data volume to 2D slices
            label_slices = self.convert_from_3d_to_2d(label_volume, True)  # Convert label volume to 2D slices
            # Iterate through each pair of data and label slices
            for data_slice, label_slice in tqdm(zip(data_slices, label_slices)):
                data_patches = self.extract_patches(data_slice, patch_size, stride)  # Extract data patches
                label_patches = self.extract_patches(label_slice, patch_size, stride)  # Extract label patches
                # Iterate through each pair of data and label patches
                for data_patch, label_patch in tqdm(zip(data_patches, label_patches)):
                    # Save data patch as an image
                    data_patch_filename = os.path.join(output_data_dir, f"data_patch_{self.count}.png")
                    cv.imwrite(data_patch_filename, data_patch)
                    # Save label patch as an image
                    label_patch_filename = os.path.join(output_label_dir, f"label_patch_{self.count}.png")
                    cv.imwrite(label_patch_filename, label_patch)
                    self.count += 1

if __name__ == "__main__":
    data_dir = r'G:\python\3d images\niiTestData\data'
    label_dir = r'G:\python\3d images\niiTestData\label'
    _output_data_dir = r'G:\python\3d images\SpineTestPatches4\data'
    _output_label_dir = r'G:\python\3d images\SpineTestPatches4\label'
    start = time.time()
    preprocessor = PreProcessing()
    # Perform preprocessing and patch extraction
    preprocessor.convert_and_save_patches(
        input_data_dir=data_dir, input_label_dir=label_dir, output_data_dir=_output_data_dir,
        output_label_dir=_output_label_dir,
        patch_size=(128, 128), stride=(128, 128))
    end = time.time()
    # Display processing time
    print(f"Completed in time: {end - start} seconds")
    print(f"Start time: {start}")
    print(f"End time: {end}")
