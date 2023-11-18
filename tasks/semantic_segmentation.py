


from typing import Any, Dict, List
import torch
from tasks import task as task_lib
import util
import vocab
import random
from vis_utils import plot_semantic_segmentation
from torchvision.ops import masks_to_boxes
import cv2
from torchmetrics import Dice
import numpy as np
from einops import rearrange
from util import multiclass_dice_coeff


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = []
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0
        

    def update(self, val, n=1):
        self.val += [val]
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
        self.std = np.std(self.val)



@task_lib.TaskRegistry.register('semantic_segmentation')
class TaskSemanticSegmentation(task_lib.Task):

    def __init__(self, config):
        super().__init__(config)
        self.reset()
    
    def reset(self):
        self.meter = AverageMeter()
        #self.criterion = Dice(average='micro') 
        #self.criterion = lambda pred, y: multiclass_dice_coeff(pred, y, reduce_label=False, n_class=self.cfg['tokens_registered'])
        self.criterion = multiclass_dice_coeff
        self.meter.reset()

    def compute(self):
        return self.meter.avg


    def preprocess_target(self, out, idx=None):
        """
        Preprocess such as obtaining polygon from mask has already been done in dataset,
        the reason is that DataLoader could have multiprocess hence it should be faster.

        Args:
        out: (im, boxes, polygons) , both box and polygon unormalised, to be quantized.
        Return:
        """
        im , mask = out
        b = im.shape[0]
        im = rearrange(im, "b c (h p1) (w p2) -> b (c h w) (p1 p2)", p1=self.cfg['patch_size_sqrt'], p2=self.cfg['patch_size_sqrt'])
        mask = rearrange(mask, "b  (h p1) (w p2) -> b (h w) (p1 p2)", p1=self.cfg['patch_size_sqrt'], p2=self.cfg['patch_size_sqrt'])
        im = im.view(b, -1)
        mask = mask.view(b,-1)
        return {'image': im , 'mask': mask}


    def preprocess_batched(self, batched_examples, training, idx=None):
        """

       Args:
           batched_examples: im (b, 1, h, w), mask (b , t).
           training: bool.

       Returns:
           images: `float` of shape (bsz, h, w, c)
           input_seq: `int` of shape (bsz, seqlen).
           target_seq: `int` of shape (bsz, seqlen).
           token_weights: `float` of shape (bsz, seqlen)

        """
        #input output in megabytes
        im = batched_examples['image']
        b = im.shape[0]
        #if im.max().int() <= 1:
        #    im = (im * 255).int()
        #weights =  (1/ batched_examples['mask'].unique(return_counts=True)[1])/(1/ batched_examples['mask'].unique(return_counts=True)[1])
        #token_weights = weights[batched_examples['mask']]


        response_seq = batched_examples['mask'] + self.cfg.segm_class_shift
        response_seq_cm = torch.zeros_like(batched_examples['mask']) # to learn encoder learn

        prompt_seq = util.build_prompt_seq_from_task_id(
            self.task_vocab_id, response_seq).to(response_seq.device)  # (bsz, 1)

        # prompt token
        response_seq_cm = torch.cat([prompt_seq, response_seq_cm], -1).long()[:,:-1]
        token_weights = torch.ones_like(response_seq)

        return im, response_seq_cm, response_seq, token_weights

        #if training:
        #  return batched_examples['image'], input_seq, target_seq, token_weights
        #else:
        #  return batched_examples['image'], response_seq, batched_examples




    def postprocess(self, im, target, logits ,dataset_obj, fname='images_train.jpg', save=True, idx=None):
        img_size = dataset_obj.cfg['train']['size']
        label_names = [name.split('/')[-1] for name in dataset_obj.cfg['train']['label']]

        if logits.ndim > 2:
            pred_seq = torch.argmax(torch.softmax(logits.permute(0,2,1),-1),-1).cpu() #logits -> [bsz seqlen token]
        else:
            pred_seq = logits.cpu()  # logits is already pred_seq
        b = pred_seq.shape[0]

        pred_seq = rearrange(pred_seq, "b (k p) -> b k p", p=self.cfg['patch_size_sqrt']**2)
        pred_seq = rearrange(pred_seq, "b (h w) (p1 p2) -> b (h p1) (w p2)", h = np.sqrt(pred_seq.shape[1]).astype(int), p1=self.cfg['patch_size_sqrt'], p2=self.cfg['patch_size_sqrt'])
        pred_seq = pred_seq.view(b, img_size, img_size)
        target = target.cpu()

        if (self.cfg['tokens_registered'] is not None) and (self.cfg.segm_class_shift != 0):
            pred_seq = torch.minimum(torch.maximum(pred_seq - self.cfg.segm_class_shift,torch.tensor(0)), torch.tensor(self.cfg['tokens_registered']-1))
        result = self.criterion(pred_seq, target, reduce_label=False, n_class=self.cfg['tokens_registered'])
        self.meter.update(result,target.shape[0])
        
        if len(im.shape) == 2:
            im = rearrange(im, "b (k p) -> b k p", p=self.cfg['patch_size_sqrt']**2)
            im = rearrange(im, "b (h w) (p1 p2) -> b (h p1) (w p2)", h = np.sqrt(im.shape[1]).astype(int), p1=self.cfg['patch_size_sqrt'], p2=self.cfg['patch_size_sqrt'])
            im = im.view(b,1, img_size, img_size)


        plots = None
        if save:
            plots = plot_semantic_segmentation(im, pred_seq, target, label_names = label_names, fname=fname, save=save)
        return {'result':result, 'plot': plots}


    def make_targets(self, pred_box, pred_cls, pred_score, device, score_thres=0.25):
        target = []
        for b,c,s in zip(pred_box, pred_cls, pred_score):
            ind = torch.where(s > score_thres)
            target += [{
                  "boxes" : b[ind].to(device),
                  "scores": s[ind].to(device),
                  "labels": c[ind].to(device)

                    }]
        return target

    def log_result(self, result,dataset_obj, mode='train'):
        out = dict([(f"{mode}_semantic_dice_{n}", result[i]) for i,n in enumerate(['background']+[name.split('/')[-1] for name in dataset_obj.cfg['train']['label']])])
        return out







