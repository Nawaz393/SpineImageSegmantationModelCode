import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class SpineDataset(Dataset):
    def __init__(self, root_path, test=False):
        """
        Initialize the SpineDataset.

        Args:
        - root_path (str): Root directory path containing 'data' and 'label' subdirectories.
        - test (bool): Flag indicating if the dataset is for testing.

        """
        self.root_path = root_path
        if not test:
            # Load images and masks for training/validation
            self.images = sorted([os.path.join(root_path, "data", i)
                                 for i in os.listdir(os.path.join(root_path, "data"))])
            self.masks = sorted([os.path.join(root_path, "label", i)
                                for i in os.listdir(os.path.join(root_path, "label"))])
        # else:
        #     # Load images and masks for testing
        #     self.images = sorted([os.path.join(root_path, "sdata", i) for i in os.listdir(os.path.join(root_path, "sdata"))])
        #     self.masks = sorted([os.path.join(root_path, "slabel", i) for i in os.listdir(os.path.join(root_path, "slabel"))])

        # Define transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset.

        Args:
        - index (int): Index of the item to retrieve.

        Returns:
        - tuple: A tuple containing the transformed image and mask.
        """
        img = Image.open(self.images[index]).convert(
            "L")  # Load and convert image to grayscale
        mask = Image.open(self.masks[index]).convert(
            "L")  # Load and convert mask to grayscale

        return self.transform(img), self.transform(mask)

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
        - int: Number of samples in the dataset.
        """
        return len(self.images)
