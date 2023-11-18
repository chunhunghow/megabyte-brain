
# Difference between instance_segmentation and semantic_segmentation in tasks
# instance_segmentation has a mask and box, while semnatic only has mask
# polygon_instance_segmentation has mask, box and polygaon


from typing import Any, Dict, List
import torch
from tasks import task as task_lib
import util
import vocab
import random
from data.data_utils import augment_bbox, jitter_bbox
from vis_utils import  plot_images_with_polygon
from torchvision.ops import masks_to_boxes
import cv2
from torchmetrics import Dice
import numpy as np
from einops import rearrange
from vis_utils import plot_semantic_segmentation
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
        self.target_sum = []
        

    def update(self, val, n=1, target_sum=0):
        self.target_sum = np.append(self.target_sum,target_sum)
        self.val += [val]
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
        self.std = np.std(self.val)



@task_lib.TaskRegistry.register('instance_segmentation')
class TaskInstanceSegmentation(task_lib.Task):

    def __init__(self, config):
        super().__init__(config)
        self.reset()
    
    def reset(self):
        self.meter = AverageMeter()
        #self.criterion = Dice(average='micro')
        #self.criterion = lambda pred,y : multiclass_dice_coeff(pred, y, reduce_label=False)
        self.criterion = multiclass_dice_coeff
        self.meter.reset()

    def compute(self):
        return self.meter.avg


    def preprocess_target(self, out, object_order='random', idx=None):
        """
        Preprocess such as obtaining polygon from mask has already been done in dataset,
        the reason is that DataLoader could have multiprocess hence it should be faster.

        Args:
        out: (im, boxes, polygons) , both box and polygon unormalised, to be quantized.
        Return:
        """
        im ,mask, boxes = out
        b = im.shape[0]
        im = rearrange(im, "b c (h p1) (w p2) -> b (c h w) (p1 p2)", p1=self.cfg['patch_size_sqrt'], p2=self.cfg['patch_size_sqrt'])
        mask = rearrange(mask, "b  (h p1) (w p2) -> b (h w) (p1 p2)", p1=self.cfg['patch_size_sqrt'], p2=self.cfg['patch_size_sqrt'])
        mask = mask.to(im.device)
        im = im.view(b, -1)
        mask = mask.view(b,-1)
        
        return {'image': im, 'mask': mask , 'boxes':boxes}


    def preprocess_batched(self, batched_examples, training, idx=None):
        """
        1. Augment the box with jitter.
        2. Quantize the box and polygons. In polygons, separator is set to >1.


       Args:
           batched_examples: im, masks, boxes
           training: bool.

       Returns:
           images: `float` of shape (bsz, h, w, c)
           input_seq: `int` of shape (bsz, seqlen). This is the bounding box seqeunce
           target_seq: `int` of shape (bsz, seqlen). This is the mask flattened sequence.
           token_weights: `float` of shape (bsz, seqlen)

        """
        config = self.cfg 
        ret = build_response_seq_for_seg(
                batched_examples['mask'], batched_examples['boxes'], 
                config.quantization_bins, config.coord_vocab_shift, config.segm_class_shift)

        response_seq, response_seq_cm, token_weights = ret
        prompt_seq = util.build_prompt_seq_from_task_id(
            self.task_vocab_id, response_seq).to(response_seq.device)  # (bsz, 1)

        # prompt token
        input_seq = torch.cat([prompt_seq, response_seq_cm], -1).long()
        target_seq = response_seq
        #target_seq = torch.cat([prompt_seq, response_seq], -1).long()
        
        #padding
        input_seq = torch.nn.functional.pad(input_seq, (0, target_seq.shape[1] - input_seq.shape[1] ) )


        if training:
          return batched_examples['image'], input_seq, target_seq, token_weights
        else:
          return batched_examples['image'], response_seq, batched_examples




    def postprocess(self, im, target, logits ,dataset_obj, fname='images_train.jpg', save=True, idx=None):

        img_size = dataset_obj.cfg.img_size
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
        result = self.criterion(pred_seq, target, reduce_label=False)
        self.meter.update(result ,target.shape[0], target.sum((1,2)))
        if len(im.shape) == 2:
            im = rearrange(im, "b (k p) -> b k p", p=self.cfg['patch_size_sqrt']**2)
            im = rearrange(im, "b (h w) (p1 p2) -> b (h p1) (w p2)", h = np.sqrt(im.shape[1]).astype(int), p1=self.cfg['patch_size_sqrt'], p2=self.cfg['patch_size_sqrt'])
            im = im.view(b,1, img_size, img_size)


        label_names = ['label0']
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
        out = dict([(f"{mode}_instance_dice_{n}", result[i]) for i,n in enumerate(['background','label0'])])
        # small lesion according to 25th percentile in INSTANCE 537, 75th 3835
        # we divide 4 because from 512 -> 256
        small = np.array(self.meter.val)[np.where(np.array(self.meter.target_sum) <= 537/4 )[0]]
        medium = np.array(self.meter.val)[(np.where((np.array(self.meter.target_sum) > 537/4) & (np.array(self.meter.target_sum) <= 3835/4 ) )[0])]
        large = np.array(self.meter.val)[np.where(np.array(self.meter.target_sum) > 3835/4 )[0]]
        out.update({f'{mode}_instance_dice_small':small.mean(0)[1],f'{mode}_instance_dice_small_n':small.shape[0] })
        out.update({f'{mode}_instance_dice_medium':medium.mean(0)[1], f'{mode}_instance_dice_medium_n':medium.shape[0]})
        out.update({f'{mode}_instance_dice_large':large.mean(0)[1], f'{mode}_instance_dice_large_n':large.shape[0]})

        return out


def build_response_seq_for_seg(mask,
                               bbox, 
                               quantization_bins,
                               coord_vocab_shift,
                               segm_class_shift):

    """
    Input sequence will be bounding box tokens (shifted). padded the rest with zeros.
    Target sequence will be flatten mask, with its label (0,1,2...) shifted by segm_class_shift
    """
    bbox = jitter_bbox(bbox)
    quantized_bbox = util.quantize(bbox, quantization_bins)
    quantized_bbox = quantized_bbox + coord_vocab_shift
    #sep_ind = (polygon > 1.)
    #is_padding = polygon == 0
    #quantized_polygon = util.quantize(polygon, quantization_bins)
    #quantized_polygon = quantized_polygon + coord_vocab_shift
    #quantized_polygon = torch.where(sep_ind, torch.zeros_like(quantized_polygon) + vocab.SEPARATOR_TOKEN, quantized_polygon)
    #quantized_polygon = torch.where(is_padding, torch.zeros_like(quantized_polygon), quantized_polygon)


    #response_seq = torch.cat([quantized_bbox, quantized_polygon],1)
    #response_seq_cm = torch.cat([quantized_bbox, quantized_polygon],1)

    #token_weights = 1. -( response_seq == 0).float()
    #token_weights[..., :quantized_bbox.shape[1]] = 0.

    #only for binary
    response_seq = mask + segm_class_shift
    response_seq_cm = quantized_bbox

    dev = response_seq.device
    token_weights = torch.where(response_seq > 0 , torch.tensor(1).to(dev), torch.tensor(0.01).to(dev))
    

    return response_seq , response_seq_cm , token_weights
    

    
    

        

