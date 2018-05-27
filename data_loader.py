import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def image_transforms(mode = 'train', tensor_type = 'torch.cuda.FloatTensor', augment_parameters = [0.8, 1.2, 0.5, 2.0, 0.8, 1.2], do_augmentation = True, transformations = None):
    if mode == 'train':
        data_transform = transforms.Compose([
            ResizeImage(),
            RandomFlip(do_augmentation),
            ToTensor(tensor_type),
            AugmentImagePair(tensor_type, augment_parameters, do_augmentation)
        ])
        return data_transform
    elif mode == 'test':
        data_transform = transforms.Compose([
            ResizeImage(False),
            ToTensor(False),
            DoTest(tensor_type),
        ])
        return data_transform
    elif mode == 'custom':
        data_transform = transforms.Compose(transformations)
        return data_transform
    else:
        return 'Wrong mode'
    
    
class ResizeImage(object):

    def __init__(self, train = True):
        self.train = train
        self.transform = transforms.Resize((256, 512))
        
    def __call__(self, sample):
        if self.train:
            left_image = sample['left_image']
            right_image = sample['right_image']
            new_right_image = self.transform(right_image)
            new_left_image = self.transform(left_image)
            sample = {'left_image': new_left_image, 'right_image': new_right_image}
        else:
            left_image = sample
            new_left_image = self.transform(left_image)
            sample = new_left_image
        return sample
    

class DoTest(object):
    
    def __init__(self, tensor_type):
        self.tensor_type = tensor_type
        
    def __call__(self, sample):
        
        new_sample = torch.from_numpy(np.stack((sample, np.fliplr(sample)), 0)).type(self.tensor_type)
        return new_sample    

    
class ToTensor(object):

    def __init__(self, train = True, tensor_type = 'torch.cuda.FloatTensor'):
        self.train = train
        self.transform = transforms.ToTensor()
        self.tensor_type = tensor_type
        
    def __call__(self, sample):
        if self.train:
            left_image = sample['left_image']
            right_image = sample['right_image']
            new_right_image = self.transform(right_image).type(self.tensor_type)
            new_left_image = self.transform(left_image).type(self.tensor_type)
            sample = {'left_image': new_left_image, 'right_image': new_right_image}
        else:
            left_image = sample
            new_left_image = self.transform(left_image).type(self.tensor_type)
            sample = new_left_image
        return sample
    
    
class RandomFlip(object):

    def __init__(self, do_augmentation):
        
        self.transform = transforms.RandomHorizontalFlip(p=1)
        self.do_augmentation = do_augmentation
        
    def __call__(self, sample):
        left_image = sample['left_image']
        right_image = sample['right_image']
        k = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if k > 0.5:
                fliped_left = self.transform(left_image)
                fliped_right = self.transform(right_image)
                sample = {'left_image': fliped_left, 'right_image': fliped_right}
        else:
            sample = {'left_image': left_image, 'right_image': right_image}
        return sample
    
    
class AugmentImagePair(object):
 
    def __init__(self, tensor_type, augment_parameters,
                 do_augmentation):
        self.tensor_type = tensor_type
        self.do_augmentation = do_augmentation
        self.gamma_low = augment_parameters[0] #0.8
        self.gamma_high = augment_parameters[1] #1.2
        self.brightness_low = augment_parameters[2] #0.5
        self.brightness_high = augment_parameters[3] #2.0
        self.color_low = augment_parameters[4] #0.8
        self.color_high = augment_parameters[5] #1.2

    def __call__(self, sample):
        left_image = sample['left_image'].type(self.tensor_type)
        right_image = sample['right_image'].type(self.tensor_type)
        p = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if p > 0.5:
                # randomly shift gamma
                random_gamma = torch.from_numpy(np.random.uniform(self.gamma_low, self.gamma_high, 1)).type(self.tensor_type)
                left_image_aug  = left_image  ** random_gamma
                right_image_aug = right_image ** random_gamma

                # randomly shift brightness
                random_brightness =  torch.from_numpy(np.random.uniform(self.brightness_low, self.brightness_high, 1)).type(self.tensor_type)
                left_image_aug  =  left_image_aug * random_brightness
                right_image_aug = right_image_aug * random_brightness

                # randomly shift color
                random_colors =  torch.from_numpy(np.random.uniform(self.color_low, self.color_high, 3)).type(self.tensor_type)
                white = torch.ones([np.shape(left_image)[1], np.shape(left_image)[2]]).type(self.tensor_type)
                color_image = torch.stack([white * random_colors[i] for i in range(3)], dim=0)
                left_image_aug  *= color_image
                right_image_aug *= color_image

                # saturate
                left_image_aug  = torch.clamp(left_image_aug,  0, 1)
                right_image_aug = torch.clamp(right_image_aug, 0, 1)

                sample = {'left_image': left_image_aug, 'right_image': right_image_aug}

        else:
            sample = {'left_image': left_image, 'right_image': right_image}
        return sample
    
    
class ImageLoader(Dataset):
    def __init__(self, root_dir, training, transform = None):
        left_dir = os.path.join(root_dir, 'left')
        self.left_paths = sorted([os.path.join(left_dir, fname) for fname\
                           in os.listdir(left_dir)])
        if training:
            right_dir = os.path.join(root_dir, 'right')
            self.right_paths = sorted([os.path.join(right_dir, fname) for fname\
                                in os.listdir(right_dir)])
            assert len(self.right_paths) == len(self.left_paths)
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        if self.training:
            right_image = Image.open(self.right_paths[idx])
            sample = {'left_image': left_image, 'right_image': right_image}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            if self.transform:
                left_image = self.transform(left_image)
            return left_image


class KittiLoader(Dataset):
    def __init__(self, root_dir, training, transform = None):
        left_dir = os.path.join(root_dir, 'image_02/data/')
        self.left_paths = sorted([os.path.join(left_dir, fname) for fname\
                           in os.listdir(left_dir)])
        if training:
            right_dir = os.path.join(root_dir, 'image_03/data/')
            self.right_paths = sorted([os.path.join(right_dir, fname) for fname\
                                in os.listdir(right_dir)])
            assert len(self.right_paths) == len(self.left_paths)
        self.transform = transform
        self.training = training


    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        left_image = Image.open(self.left_paths[idx])
        if self.training:
            right_image = Image.open(self.right_paths[idx])
            sample = {'left_image': left_image, 'right_image': right_image}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            if self.transform:
                left_image = self.transform(left_image)
            return left_image
