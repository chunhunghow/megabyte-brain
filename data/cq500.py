


import os
import torch
from torch.utils.data import DataLoader
from data import dataset as dataset_lib
import sys
from pathlib import Path
sys.path.insert(1, f'{Path(os.getcwd()).parent}/pseudohealthy')
from utils.dataloaders import LoadImagesAndLabels
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective, PixelShuffling
import re
import pandas as pd
import numpy as np
import random


class MidLineDataset(LoadImagesAndLabels):
    '''
    CQ500 image level labels
    '''
    

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        ###=========================================
        #      Chun Hung added lines to map Scan ID 
        ###=========================================
        
        patient = re.search('CQ500-Unzip_([0-9]+)',self.im_files[index]).group().split('CQ500-Unzip_')[-1]
        multilabels = self.cq500_labels.query("id == @patient")[self.TARGET_LABELS].values # assigned in this script 
        multilabels = torch.tensor(multilabels)
        ###=========================================

    
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None
    
            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))
    
        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)
    
            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
    
            labels = self.labels[index].copy()
            #if labels.size:  # normalized xywh to pixel xyxy format
            #    labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
    
            if self.augment:
                img, _       = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])
    
    
        if self.augment:
            # Albumentations
            img, _ = self.albumentations(img, labels)
    
            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
    
            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)


        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(img)
        im = (im - im.min())/np.ptp(im) 
        im = torch.from_numpy(im).to(torch.float32)
        return im , multilabels[0]  




def collate_fn_bhx(batch):
    ori, ori_bbox = zip(*batch)  # transposed
    for i, lb in enumerate(ori_bbox):
        lb[:, 0] = i  # add target image index for build_targets()
    

    return torch.stack(ori, 0), torch.cat(ori_bbox, 0)


@dataset_lib.DatasetRegistry.register('cq500_classification')
class CQ500Dataset(dataset_lib.Dataset):

    """
    Require TARGET_LABELS if dataset is classification

    """
    def __init__(self, config):
        self.cfg = config
        cq500_labels = config.label_path
        cq500_labels = pd.read_csv(cq500_labels)
        cq500_labels['id'] = cq500_labels['name'].apply(lambda x: x.split('CQ500-CT-')[-1])
        TARGET_LABELS = [col[3:] for col in cq500_labels.columns if (np.isin(cq500_labels[col].unique(), [0, 1]).all() & ("R1" in col)) ]
        for col in TARGET_LABELS:
            col_sum = (cq500_labels['R1:'+ col].add(cq500_labels['R2:'+ col]).add(cq500_labels['R3:'+col]) >1 ).astype(int)
            cq500_labels[col] = col_sum
        self.cq500_labels = cq500_labels
        TARGET_LABELS.remove('ICH')
        self.TARGET_LABELS = TARGET_LABELS


    def load_dataset(self, mode):

        augment = True if mode == 'train' else False
        hyp = {
                'mosaic': 0.0,
                'hsv_h' : 0,
                'hsv_s': 0,
                'hsv_v': 0,
                'flipud' : 0.2,
                'fliplr' : 0.5,
                'degrees' : 10,
                'translate':.1,
                'scale':0.2,
                'shear':5,
                'perspective': 0,
        
                } # if we want augment, has to include hypa
        assert mode in ["train", 'val', 'test'] 
        dataset = MidLineDataset(self.cfg[f'{mode}_path'], img_size=self.cfg['img_size'], batch_size=self.cfg['batch_size'], augment= augment, hyp=hyp)
        dataset.cq500_labels = self.cq500_labels
        dataset.TARGET_LABELS = self.TARGET_LABELS

        return dataset


    def load_dataloader(self,mode ):

        dataset = self.load_dataset(mode)

        loader = DataLoader(
            dataset,
            #sampler=train_sampler if 'cq500' in opt.data.lower() else None,
            sampler = None,
            batch_size= self.cfg['batch_size'],
            num_workers=self.cfg.num_workers,
            collate_fn = None,
            shuffle= True,
            drop_last=False)

        return loader

    def process(self, batch, **kwargs):
        im,target = batch
        if im.shape[1] > 1:
            im = im[:,:1,...]

        return im, target


