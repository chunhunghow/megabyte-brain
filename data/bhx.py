


import os
import torch
from torch.utils.data import DataLoader
from data import dataset as dataset_lib
import sys
from pathlib import Path
sys.path.insert(1, f'{Path(os.getcwd()).parent}/pseudohealthy')
from utils.dataloaders import LoadImagesAndLabels


def collate_fn_bhx(batch):
    ori, ori_bbox = zip(*batch)  # transposed
    for i, lb in enumerate(ori_bbox):
        lb[:, 0] = i  # add target image index for build_targets()
    

    return torch.stack(ori, 0), torch.cat(ori_bbox, 0)


@dataset_lib.DatasetRegistry.register('bhx_detection')
class BHXDataset(dataset_lib.Dataset):
    def __init__(self, config):
        self.cfg = config


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
        #hyp = {
        #        'mosaic': 0.0,
        #        'hsv_h' : 0,
        #        'hsv_s': 0,
        #        'hsv_v': 0,
        #        'flipud' : 0.0,
        #        'fliplr' : 0.0,
        #        'degrees' : 0,
        #        'translate':.0,
        #        'scale':0.0,
        #        'shear':0,
        #        'perspective': 0,
        #
        #        } # if we want augment, has to include hypa
        assert mode in ["train", 'val', 'test'] 
        dataset = LoadImagesAndLabels(self.cfg[f'{mode}_path'], img_size=self.cfg['img_size'], batch_size=self.cfg['batch_size'], augment= augment, hyp=hyp)

        return dataset


    def load_dataloader(self,mode ):

        dataset = self.load_dataset(mode)

        loader = DataLoader(
            dataset,
            #sampler=train_sampler if 'cq500' in opt.data.lower() else None,
            sampler = None,
            batch_size= self.cfg['batch_size'],
            num_workers=self.cfg.num_workers,
            collate_fn = collate_fn_bhx,
            shuffle= True,
            drop_last=False)

        return loader


    def process(self, batch, target_format='xyxyn', idx=None):
        """
        Args:
            target_format: Transform the format of the box at output, input box assumed to have cxcywhn
        """
        im,bbox = batch
        if im.shape[1] > 1:
            im = im[:,:1,...]
        target = []
        for i in range(im.shape[0]):
            dic = {}
            dic['labels'] = bbox[bbox[:,0] == i,1].to(int).cuda()
            b = bbox[bbox[:,0] == i,2:].cuda() #cxcywhn 
            b[:,0] = b[:,0]-b[:,2]/2  #get xyxyn
            b[:,1] = b[:,1]-b[:,-1]/2  
            b[:,2] = b[:,0]+b[:,2]
            b[:,3] = b[:,1]+b[:,-1]


            if target_format == 'xyxyn':
                pass
            elif target_format == 'xyxy':
                b[:,0] *= im.shape[-1] #w
                b[:,1] *= im.shape[-2] #h
                b[:,2] *= im.shape[-1] #w
                b[:,3] *= im.shape[-2] #h
            else:
                raise NotImplementedError(f'{target_format}')
            dic['boxes'] = b
            target += [dic]

        return (im, target)
