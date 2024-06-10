import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class FaceDataset(Dataset):
    def __init__(self, labels, img_dir, transform=None):
        self.input_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len([name for name in os.listdir(self.input_dir) if ".jpg" in name])
    
    def __getitem__(self, idx):
        if self.transform is None:
            tfs = transforms.Compose([transforms.ConvertImageDtype(torch.float),
                                      transforms.Resize(64),
                                      transforms.CenterCrop(64),
                                      transforms.Normalize((0.4, 0.4, 0.4), (0.4, 0.4, 0.4))])   
                    
            sample = tfs(read_image(self.input_dir+f"/{idx+1:06d}"+".jpg"))
            
            return sample, []