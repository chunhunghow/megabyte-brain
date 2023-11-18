

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import os
import imageio
import numpy as np
from PIL import Image
import cv2
from einops import rearrange
from data import dataset as dataset_lib

class LoadTissue:
    def __init__(self,input, label, size, split=None,augment=False ,**kwargs):
        #self.data_cfg = config
        self.input_path = input
        self.class_target = label
        self.augment = augment
        self.input_images = [f for f in os.listdir( self.input_path ) if f.endswith(".tiff")]
        if split is not None:
            assert isinstance(split, list), "`split` must be a tuple of data split float, such as (0.8,0.9) that will be 80%-90% of the data, or [0 , 0.8]"
            assert len(split) == 2
            assert split[0] < split[1]
            print(f"Splitting data as {split}")
            self.input_images = self.input_images[int(split[0] * len(self.input_images)) : int(split[1] * len(self.input_images))]

                


        self.img_size = size

        self.transform_labeled = A.Compose([
 #                         A.ToGray(p=0.5),
                          A.RandomGamma(p=0.3),
                          A.HorizontalFlip(p=0.5),
                          A.augmentations.geometric.transforms.Affine(scale=1.2,rotate=(-15,15),shear=(15,15),),
                          A.RandomResizedCrop(height=size,
                                            width=size,
                                            scale=(0.8,1.0),
                                            p=1),
                          ToTensorV2(),])

    def get_data(self, name):
        x = np.array(imageio.imread(Path(self.input_path) / name))
        path_y = [np.zeros_like(x)[..., np.newaxis] ] #background class
        for c in self.class_target:
            path_y += [ np.array(Image.open(Path(c) / str(name.split('.')[0] + '.png' ) ))[ ..., np.newaxis] ]

        mask = np.concatenate(path_y, -1)
        h0, w0 = x.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if (r != 1) or (h0 != w0):  # if sizes are not equal
            interp = cv2.INTER_LINEAR if self.augment else cv2.INTER_AREA  # random.choice(self.rand_interp_methods)
            x = cv2.resize(x, (self.img_size, self.img_size), interpolation=interp)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=interp)

        #data is flipped
        x = np.flip(rearrange(x, 'h w -> w h'), axis=-2).copy() # image is 2D
        mask = np.flip(rearrange(mask,'h w c -> w h c' ), axis=0).copy() # channel at last dim

        return x, mask



        
        
    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        im , mask = self.get_data(self.input_images[idx])
        if self.augment:
            augmented = self.transform_labeled(image=im, mask=mask)
            return augmented['image'].to(torch.float), rearrange(augmented['mask'].to(torch.float), 'h w c -> c h w')

        else:
            return torch.from_numpy(im).to(torch.float)[None,], rearrange(torch.from_numpy(mask.copy()), 'h w c -> c h w')



@dataset_lib.DatasetRegistry.register('tissue_segmentation')
class TissueDataset():
    def __init__(self, cfg):
        self.cfg = cfg

    def load_dataset(self, mode='train'):
        return LoadTissue(**self.cfg[mode])

    def load_dataloader(self, mode='train'):
        loader = torch.utils.data.DataLoader(
            self.load_dataset(mode),
            sampler = None,
            batch_size= self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn = None,
            shuffle= True,
            drop_last=False)

        return loader

    def process(self, out, idx=None):
        im, mask = out
        if mask is not None:
            mask = torch.argmax(mask,1)
        else:
            mask = mask
        return im,mask



import pandas as pd
class LoadSingleScan:
    def __init__(self,input, label, size, batch_size=None, augment=False ,**kwargs):
        #self.data_cfg = config
        self.input_path = input
        self.class_target = label
        self.augment = augment
        self.input_images = pd.Series(os.listdir( self.input_path ))
        self.scanID = np.unique([s.split('_')[0] for s in self.input_images])
        self.img_size = size
        self.batch_size = batch_size

        self.transform_labeled = A.Compose([
 #                         A.ToGray(p=0.5),
                          A.RandomGamma(p=0.3),
                          A.HorizontalFlip(p=0.5),
                          A.augmentations.geometric.transforms.Affine(scale=1.2,rotate=(-15,15),shear=(15,15),),
                          A.RandomResizedCrop(height=size,
                                            width=size,
                                            scale=(0.8,1.0),
                                            p=1),
                          ToTensorV2(),])

    def get_data(self, name):
        x = np.array(imageio.imread(Path(self.input_path) / name))
        path_y = [np.zeros_like(x)[..., np.newaxis] ] #background class
        for c in self.class_target:
            path_y += [ np.array(Image.open(Path(c) / str(name.split('.')[0] + '.png' ) ))[ ..., np.newaxis] ]

        mask = np.concatenate(path_y, -1)
        h0, w0 = x.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if (r != 1) or (h0 != w0):  # if sizes are not equal
            interp = cv2.INTER_LINEAR if self.augment else cv2.INTER_AREA  # random.choice(self.rand_interp_methods)
            x = cv2.resize(x, (self.img_size, self.img_size), interpolation=interp)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=interp)

        #data is flipped
        x = np.flip(rearrange(x, 'h w -> w h'), axis=-2).copy() # image is 2D
        mask = np.flip(rearrange(mask,'h w c -> w h c' ), axis=0).copy() # channel at last dim

        return x, mask

        
    def __len__(self):
        return len(self.scanID)

    def __getitem__(self, idx):
        scanid = self.scanID[idx]
        im_list = self.input_images[self.input_images.str.contains(scanid)].values
        ims = []
        masks = [] 
        np.random.shuffle(im_list)
        if self.batch_size is not None:
            im_list = im_list[:self.batch_size]
        for s in im_list:
            im , mask = self.get_data(s)
            if self.augment:
                augmented = self.transform_labeled(image=im, mask=mask)
                ims += [augmented['image'].to(torch.float)]
                masks += [augmented['mask'].to(torch.float)]
            else:
                ims += [torch.from_numpy(im).to(torch.float)[None,]]
                masks += [rearrange(torch.from_numpy(mask.copy()), 'h w c -> c h w')[None,]]
         
        return torch.cat(ims) , torch.cat(masks)
