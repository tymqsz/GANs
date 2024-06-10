import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms

class MazesDataset(Dataset):
    def __init__(self, labels, img_dir, transform=None):
        self.input_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len([name for name in os.listdir(self.input_dir) if ".png" in name])
    
    def __getitem__(self, idx):
        if self.transform is None:   
            sample = read_image(self.input_dir+"/"+str(idx)+".png")
            
            return sample, []