



from data import dataset as dataset_lib
import os
import glob
import pydicom
import torch
from torch.utils.data import Dataset
import albumentations as A
import nrrd
import sys
from albumentations.pytorch import ToTensorV2
import re
import io
from PIL import Image
import numpy as np
import cv2
import random
from torchvision.ops import masks_to_boxes
import vocab

class LoadPhysioNet():

    '''
    PhysioNet, INSTANCE dataset has been stored into PNG
    Turn mask into polygon contour.

    '''
    def __init__(self, 
                 path=None, 
                 mode='abnormal', 
                 label=False, 
                 img_size=256, 
                 max_seq_len =512 , 
                 rotate_polygon = True,
                 eps = 0.001,
                 healthy_path = '/home/data_repo/physionetICH/data2D/healthy'):
        '''
        Args:
        path: path to folder of abnormal 
        max_seq_len: The sequence length for each polygon 
        eps: epsilon distance for rendering cv2 contour.
        rotate_polygon: Augment the contour points, any point can be starting point.
        '''
        assert mode in ['abnormal', 'normal', 'all'] , 'Mode should be `abnormal`, `normal` or `all`'
        self.mode = mode
        self.img_size = img_size
        self.label = label
        self.train = True if 'train' in path else False
        healthy_images = glob.glob(healthy_path+ '/*')
        if self.mode == 'all':
            self.images = glob.glob(path + '/*')
            self.images = healthy_images + self.images
        elif self.mode == 'normal':
            self.images = healthy_images
        else:
            self.images = glob.glob(path + '/*')
        self.transform_labeled = A.Compose([
              A.HorizontalFlip(p=0.5),              
              A.Affine(scale=(0.90,1.0),
                       translate_percent=(0.,0.2),
                       rotate=(-15,15),
                       shear=(-8,8)
                  ),
              #A.RandomResizedCrop(height=img_size,
              #                    width=img_size, 
              #                    scale=(0.85,1.0),
              #                    p=0.7),
              A.RandomBrightnessContrast(),
              A.GaussNoise(),
              ToTensorV2(), #not normalized
              ])


        #self.aug1 = PixelShuffling(30)
        self.max_seq_len = max_seq_len
        self.rotate_polygon = rotate_polygon
        self.eps = eps

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        """
        Actual function to load public annotated dataset. 

        Returns:
            im : `Tensor`
            boxes: `Tensor` (n 4) Absolute coordinates.
            contours: `List[Tensor]` A of list of lists of polygon coordinates.
            where tensor in shape ( 256, 1, 3) where 3 refers to (x, y, contour_ind) , one box can contain more than 1 contour.
            List is the number of boxes.
                    
        """

        im = np.array(Image.open(self.images[idx])) #[h,w,3]
        if len(im.shape) == 3:
            im = im[:,:,0]

        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_AREA  # random.choice(self.rand_interp_methods)
            im = cv2.resize(im, (self.img_size, self.img_size), interpolation=interp)           

        if (self.mode == 'abnormal') & self.label:
            p = re.sub( 'images','masks',self.images[idx])
            mask = np.array(Image.open(p))

            if r != 1:
                interp = cv2.INTER_AREA  # random.choice(self.rand_interp_methods)
                mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=interp)           
            mask = (mask >0).astype(int)

            assert mask.sum() > 0 , f'Empty Mask before transform! {self.images[idx]} {mask.shape}'
            if self.train:
                pass_  = 0
                while pass_ == 0:
                    augmented = self.transform_labeled(image=im, mask=mask)
                    mask_attempt = augmented['mask']
                    if mask_attempt.sum() > 0 : 
                        pass_ = 1
                        im = augmented['image']
                        mask = augmented['mask']
            else:
                #im = torch.tensor(im).permute(2,0,1)
                im = torch.tensor(im)
                mask = torch.tensor(mask)
            im = (im - im.min())/np.ptp(im)

            # Knowing that the mask is assumed to be single lesion only.
            assert mask.sum() > 0 , f'Empty Mask after transform! {self.images[idx]} {mask.shape}'
            boxes = masks_to_boxes(mask[None,]) #xyxy

            if len(boxes) == 0:
                nz = mask.nonzero()
                xmin , ymin = nz[:,1].min() -2 , nz[:,0].min() -2
                xmax, ymax = max(nz[:,1].max() +2, xmin+2) , max(nz[:,0].max()+2, ymin+2)
                boxes = torch.tensor([[xmin,ymin, xmax, ymax]])
                assert len(boxes) > 0 

            #contours_list = []
            #for i , box in enumerate(boxes):
            #    m = torch.zeros_like(mask)
            #    box = box.to(int)
            #    #erratic mask that is vertical line for ex
            #    if box[2] - box[0] < 2:
            #        box[2] += 1
            #    
            #    if box[3] - box[1] < 2:
            #        box[3] += 1

            #    xmin, ymin, xmax, ymax = box
            #    m[ymin:ymax, xmin:xmax] = 1
            #    box_mask = m * mask
            #    assert box_mask.sum() > 0 , f"{m.sum()} {mask.sum()} {box_mask.sum()} {box}"
            #    m = (box_mask.numpy() * 255).astype(np.uint8)
            #    edged = cv2.Canny(m, 0, 255)
            #    contours, hierarchy = cv2.findContours(edged, 
            #            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #    
            #    assert len(contours) > 0 , f'masked {m.sum()} {self.images[idx]} {box}'
            #    # a full contour has too many dense points, usually the sequence is 512,
            #    # we can approximate the polygonal curve
            #    # epsilon represents the maximum distance between the approximation of a shape ..
            #    # .. contour of the input polygon and the original input polygon
            #    # so straight line will be distance 0 between orignal and approx

            #    # we do for each contour in the box, all contours in same box will be separated
            #    # by a token
            #    # paper mentioned that max point of polygon is set to 128
            #    approx_contours = []
            #    max_len_per_cnt = ((self.max_seq_len  - len(contours) -4) //2 ) // len(contours) # minus placeholder for separation token, 4 for given coordinates
            #    for cnt in contours:
            #       arclen = cv2.arcLength(cnt, True)
            #       epsilon = arclen * self.eps
            #       approx = cv2.approxPolyDP(cnt, epsilon, True)

            #       #approx (n,1,2) n is the number of points
            #       if approx.shape[0] > max_len_per_cnt:
            #           r = max_len_per_cnt / approx.shape[0] 
            #           samp = np.arange(int(r * approx.shape[0]))
            #           np.random.shuffle(samp)
            #           samp = sorted(samp[:max_len_per_cnt])
            #           approx = torch.tensor(approx[samp])
            #       else:
            #           approx = torch.tensor(approx)

            #       assert approx.shape[0] <= max_len_per_cnt, f'approx number of points {approx.shape} > {max_len_per_cnt+1}, Number of contours in this box {len(contours)}'

            #       #augmentation, we can start from any point 
            #       start_ind = np.random.randint(0,approx.shape[0])
            #       approx = torch.cat([approx[start_ind:], approx[:start_ind]])

            #       approx = approx.flatten() #(n 1 2) -> (nx2)

            #       # SEPARATOR: a token for separation, pad separator for now.
            #       approx = torch.cat([ approx, torch.tensor([self.img_size+1]) ])


            #       approx_contours += [approx]  # approx is one contour
  
            #    approx_contours = torch.cat(approx_contours)
            #    pad = (self.max_seq_len -4 ) - approx_contours.shape[0]  # 4 is for prompt token
            #    assert pad >= 0
            #    if pad > 0: # pad seq to full length due to floor division
            #        #approx_contours = torch.cat([approx_contours, torch.zeros((pad,1,2),dtype=int)],0) #3 because one dim is index
            #        approx_contours = torch.cat([approx_contours, torch.zeros(pad,dtype=int)]) #3 because one dim is index
            #    contours_list += [approx_contours]
            #    # all contours in a box, (num_cont, seqlen/2, 1,2)
            
            if self.train:
                return im, mask, boxes
            else:
                return im[None,...], mask, boxes



        im = (im - im.min())/np.ptp(im)
        return torch.from_numpy(im).to(torch.float32)[None,...]




@dataset_lib.DatasetRegistry.register('lesion_segmentation')
class InstanceSegDataset(dataset_lib.Dataset):

    """

    """
    def __init__(self, config):
        self.cfg = config

    def collate_fn(batch):
        """
        To process and sample batch_size of polygons
        contours_list : List[List[Tensor]]

        Return:
        Since standard output should be im and target, we will concat boxes and polygons, 
        then separate them again in preprocess_target fucntion. Check main.py training_steps.
        """

        ## here we just assume one image one box
        # that means boxes will have (n,4) instead of (1,4) for each box in boxes
        im, mask, boxes = zip(*batch)
        #======== check for multi boxes per image =============
        box_count = [box.shape[0] for box in boxes]
        box_choice = [np.random.randint(c) for c in box_count]
        new_boxes = []
        
        for i,box in enumerate(boxes):
            new_boxes += [box[box_choice[i]][None,]]
            #new_contours_list += [pol[box_choice[i]]]

        boxes = new_boxes
        del new_boxes
        #====================================================
        
        #========== check for contour ===================
        #ind = np.concatenate([[ci[...,0].unique().tolist() for ci in c] for c in contours_list])
        #choice = [np.random.randint(len(l)) for l in ind] # expecting some image have multiple boxes, pick just one randomly
        #polygons = torch.cat([c[choice[i]][...,1:][None,] for i,c in enumerate(contours_list)]) # remove contour index
        #polygons = torch.cat([c[...,1:][None,] for i,c in enumerate(contours_list)]) # remove contour index

        im = torch.cat([img[None,] for img in im])
        mask = torch.cat([m[None,] for m in mask])
        #polygons = torch.cat([c[None,] for c in contours_list])

        return im, mask, torch.cat(boxes)

    def load_dataset(self, mode='train'):
        assert mode in ('train', 'test', 'val')
        data_path = self.cfg[f'{mode}_path']
        dataset = LoadPhysioNet(data_path, mode='abnormal', img_size=self.cfg.img_size, label=True, max_seq_len = self.cfg.max_seq_len, eps=self.cfg.eps)      
        return dataset

    def load_dataloader(self,mode):
        dataset = self.load_dataset(mode)
        loader = torch.utils.data.DataLoader(
            dataset,
            sampler = None,
            batch_size= self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn = InstanceSegDataset.collate_fn,
            #shuffle= True,
            drop_last=False)
        return loader

    def process(self, batch, target_format='xyxyn', idx=None):
        """
        Normalised the coordinates, flatten the polygons. Input is assumed to be absolute coord. 
        If original is xyxy, then target should be xyxyn.
        In __getitem__ , separator is padded as img_size+1

        Args:
            im : in batch, (bs, c, h, w)
            boxes : in batch, (bs, 1, 4) one box from each image
            target_format: format for box and polygon (absolute coord) at output
        """
        im,mask, boxes = batch
        if im.shape[1] > 1:
            im = im[:,:1,...]

 
        if target_format == 'xyxy':
            pass

        elif target_format == 'xyxyn':
            boxes[:,0] /= im.shape[-1] #w
            boxes[:,1] /= im.shape[-2] #h
            boxes[:,2] /= im.shape[-1] #w
            boxes[:,3] /= im.shape[-2] #h
        else:
            raise NotImplementedError(f'{target_format}')

        return im, mask.cpu() ,boxes
