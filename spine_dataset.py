import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class SpineDataset(Dataset):
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        if not test:
            self.images = sorted(
                [root_path + "/data/" + i for i in os.listdir(root_path + "/data/")])
            self.masks = sorted(
                [root_path + "/label/" + i for i in os.listdir(root_path + "/label/")])
        # else:
        #     self.images = sorted([root_path+"/sdata/"+i for i in os.listdir(root_path+"/sdata/")])
        #     self.masks = sorted([root_path+"/slabel/"+i for i in os.listdir(root_path+"/slabel/")])

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        # print(self.images[index])
        # print(self.masks[index])
        img = Image.open(self.images[index]).convert("L")
        mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)
