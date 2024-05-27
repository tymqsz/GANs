import os
from torchvision.io import read_image
from torch.utils.data import Dataset

class MonetDataset(Dataset):
    def __init__(self, labels, img_dir, transform=None):
        self.input_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len([name for name in os.listdir(self.input_dir) if "jpg" in name and "d" not in name])
    
    def __getitem__(self, idx):
        if self.transform is None:
            return read_image(self.input_dir+"/"+str(idx)+".jpg"), []