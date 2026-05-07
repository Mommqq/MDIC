import os
from random import random
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from PIL import Image


class PairCityscape(Dataset):

    def __init__(self, path, set_type, resize=(128, 256)):
        super(Dataset, self).__init__()
        self.resize = resize

        self.dataset = {
            'left': os.path.join(path, 'leftImg8bit', set_type),
            'right': os.path.join(path, 'rightImg8bit', set_type)
        }

        self.cities = sorted([
            item for item in os.listdir(self.dataset['left'])
            if os.path.isdir(os.path.join(self.dataset['left'], item))
        ])

        self.ar = []
        for city in self.cities:
            left_dir = os.path.join(self.dataset['left'], city)
            right_dir = os.path.join(self.dataset['right'], city)
            pair_names = sorted([
                '_'.join(f.split('_')[:-1])
                for f in os.listdir(left_dir)
                if os.path.splitext(f)[-1].lower() == '.png'
            ])
            for pair in pair_names:
                left_img = os.path.join(left_dir, pair + '_leftImg8bit.png')
                right_img = os.path.join(right_dir, pair + '_rightImg8bit.png')
                self.ar.append((left_img, right_img))

        if set_type == 'train':
            self.transform = self.train_deterministic_cropping
        elif set_type == 'test' or set_type == 'val':
            self.transform = self.test_val_deterministic_cropping

    def train_deterministic_cropping(self, img, side_img):
        # Resize
        img = TF.resize(img, self.resize)
        side_img = TF.resize(side_img, self.resize)

        # Random Horizontal Flip
        if random() > 0.5:
            img = TF.hflip(img)
            side_img = TF.hflip(side_img)

        # Convert to Tensor

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])
        # Convert to Tensor
        # img = transforms.ToTensor()(img)
        # side_img = transforms.ToTensor()(side_img)

        img = transform(img)
        side_img = transform(side_img)
        return img, side_img

    def test_val_deterministic_cropping(self, img, side_img):
        # Resize
        img = TF.resize(img, self.resize)
        side_img = TF.resize(side_img, self.resize)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])
        # Convert to Tensor
        # img = transforms.ToTensor()(img)
        # side_img = transforms.ToTensor()(side_img)

        img = transform(img)
        side_img = transform(side_img)
        return img, side_img

    def __getitem__(self, index):
        left_path, right_path = self.ar[index]

        img = Image.open(left_path)
        side_img = Image.open(right_path)
        image_pair = self.transform(img, side_img)

        return {'img':image_pair[0], 'cor_img':image_pair[1], 'path_pair':(left_path, right_path), 'input_ids':index }

    def __len__(self):
        return len(self.ar)

    def __str__(self):
        return 'Cityscape'


if __name__ == '__main__':
    ds = PairCityscape(path='./', set_type='train')
    ds = DataLoader(dataset=ds)
    for data in ds:
        img, cor_img, idx, _ = data
        print(img.shape, idx)
    print(len(ds))
