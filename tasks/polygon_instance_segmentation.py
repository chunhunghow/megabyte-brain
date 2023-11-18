


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



@task_lib.TaskRegistry.register('instance_segmentation')
class TaskInstanceSegmentation(task_lib.Task):

    def __init__(self, config):
        super().__init__(config)
        self.reset()
    
    def reset(self):
        self.meter = AverageMeter()
        self.criterion = Dice(average='micro')
        self.meter.reset()

    def compute(self, mode='val'):
        return self.meter.avg


    def preprocess_target(self, out, object_order='random'):
        """
        Preprocess such as obtaining polygon from mask has already been done in dataset,
        the reason is that DataLoader could have multiprocess hence it should be faster.

        Args:
        out: (im, boxes, polygons) , both box and polygon unormalised, to be quantized.
        Return:
        """
        im , _ , boxes, polygons = out
        
        return {'image': im , 'boxes':boxes, 'polygons':polygons}


    def preprocess_batched(self, batched_examples, training):
        """
        1. Augment the box with jitter, polygon has been augmented (rotated) during data sampling.
        2. Quantize the box and polygons. In polygons, separator is set to >1.


       Args:
           batched_examples: im, boxes, polygons.
           training: bool.

       Returns:
           images: `float` of shape (bsz, h, w, c)
           input_seq: `int` of shape (bsz, seqlen).
           target_seq: `int` of shape (bsz, seqlen).
           token_weights: `float` of shape (bsz, seqlen)

        """
        config = self.cfg 
        ret = build_response_seq_for_seg(
                batched_examples['boxes'], batched_examples['polygons'], 
                config.quantization_bins, config.coord_vocab_shift)

        response_seq, response_seq_cm, token_weights = ret
        prompt_seq = util.build_prompt_seq_from_task_id(
            self.task_vocab_id, response_seq)  # (bsz, 1)

        # prompt token
        input_seq = torch.cat([prompt_seq, response_seq_cm], -1).long()
        target_seq = torch.cat([prompt_seq, response_seq], -1).long()
        

        #target seq will not have prompt token
        input_seq, target_seq = input_seq[..., :-1], target_seq[..., 1:]
        

        if training:
          return batched_examples['image'], input_seq, target_seq, token_weights
        else:
          return batched_examples['image'], response_seq, batched_examples




    def postprocess(self, im, target, logits ,dataset_obj, fname='images_train.jpg', save=True):
        img_size = dataset_obj.cfg.img_size
        pred_seq = torch.argmax(torch.softmax(logits.permute(0,2,1),-1),-1).cpu() #logits -> [bsz seqlen token]

        # remove first 4 steps from prediction
        pred_seq = pred_seq[:, 4:]

        pred_polygon = \
          util.decode_seq_to_polygon( pred_seq, self.cfg.quantization_bins, self.cfg.coord_vocab_shift)
        #target_polygon = \
        #  util.decode_seq_to_polygon( target, self.cfg.quantization_bins, self.cfg.coord_vocab_shift)

        mask = plot_images_with_polygon(im , pred_polygon , target, fname=fname, target_format='xyxy', save=save) #plot_image is adaptive to number of images in validation step
        result = self.criterion(mask, target)
        self.meter.update(result,target.shape[0])
        return result


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
        return {f'{mode}_Dice': result}


def build_response_seq_for_seg(bbox,
                               polygon, 
                               quantization_bins,
                               coord_vocab_shift):

    """
    For segmentation task, input prompt is task_id and 4 coordinates during inference.
    input_seq is target_seq, the sequence start with 4 box token, before adding task token.
    So if max sequence is 512, we only allow 508 to be filled by polygon coordinates.
    """

    bbox = jitter_bbox(bbox)
    quantized_bbox = util.quantize(bbox, quantization_bins)
    quantized_bbox = quantized_bbox + coord_vocab_shift
    sep_ind = (polygon > 1.)
    is_padding = polygon == 0
    quantized_polygon = util.quantize(polygon, quantization_bins)
    quantized_polygon = quantized_polygon + coord_vocab_shift
    quantized_polygon = torch.where(sep_ind, torch.zeros_like(quantized_polygon) + vocab.SEPARATOR_TOKEN, quantized_polygon)
    quantized_polygon = torch.where(is_padding, torch.zeros_like(quantized_polygon), quantized_polygon)


    response_seq = torch.cat([quantized_bbox, quantized_polygon],1)
    response_seq_cm = torch.cat([quantized_bbox, quantized_polygon],1)

    # Special trick for instance segmentation, do not require the model to learn the bbox in front of the polygon
    # will be provided as prompt during inference

    token_weights = 1. -( response_seq == 0).float()
    token_weights[..., :quantized_bbox.shape[1]] = 0.

    return response_seq , response_seq_cm , token_weights
    

    
    

        

