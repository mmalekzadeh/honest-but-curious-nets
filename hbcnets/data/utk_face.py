import os
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
import torchvision.transforms as transforms

class UTKFace(Dataset):
    r"""
    - Parts of the code from: github.com/narumiruna/UTKFace-utils
    - UTKFace dataset can be found here: https://susanqq.github.io/UTKFace
    - [age] is an integer from 0 to 116, indicating the age
    - [gender] is either 0 (male) or 1 (female)
    - [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others
    """    
    def __init__(self, image_size=64, root="UTKFace", transform=None):
        self.root = root
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(size=(image_size,image_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.mul(2).sub(1)), # From [0,1] to [-1,1]
                ])
        self.transform = transform
        self.samples = self._prepare_samples(root)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = pil_loader(path)

        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)

    def _prepare_samples(self, root):
        samples = []
        paths = Path().rglob(root+'/*.jpg')        
        paths = sorted(paths)       
        for path in paths:            
            try:
                label = self._load_label(path)
            except Exception as e:
                print('path: {}, exception: {}'.format(path, e))
                continue

            samples.append((path, label))
        return samples
    
    def _load_label(self, path):
        str_list = os.path.basename(path).split('.')[0].strip().split('_')                
        age, gender, race = map(int, str_list[:3])
        label = dict(age=age, gender=gender, race=race)
        return label