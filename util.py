

import torch
import torchvision
import copy
import vocab
from torch import Tensor
import numpy as np


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def create_mask(tgt, pad_idx):
    '''
    tgt : [n L]
    pad_idx : The token that indicates padding in the target.
    diagonal : Set to 0 , even for instance segmentation task where we provide 4 known coordinates.
               For instance segmentation task, While training, model will learn to output the prompt coordinates
               by attending to only past token, for example (task_id, x1, y1, _?_ ...), but at inference, (task_id, x1,y1,x2,y2) will be given.

    Return
        tgt_mask : Triangular matrix (seq_len, seq_len) containing -inf for future steps.
    '''

    tgt_seq_len = tgt.shape[1]
    diagonal = 0 
    tgt_mask = torch.triu(torch.ones((tgt_seq_len,tgt_seq_len)), diagonal= -diagonal).transpose(0,1) #lower triangular
    #fill 0 as -inf to prevent future
    tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, 0.0)
    ## padding mask
    tgt_padding_mask = (tgt == pad_idx)
    return tgt_mask.to(tgt.device) , tgt_padding_mask.to(tgt.device)



class Tokenizer:
    def __init__(self, bins=1000):
        self.bins = bins

    def quantize(self,x):
        '''
        x is a real number between [0, 1]
        '''
        return int(x * (bins - 1))

    def dequantize(self, x):
        return float(x) / (bins - 1)


def quantize(x, bins=1000):
    '''
    x is a tensor containing real number between [0, 1]
    '''
    coord = torch.round(x * (bins -1)).to(torch.int)
    coord = torch.clip(coord, 0,  bins-1)
    return coord

def dequantize( x, bins=1000):
    boxes = x.to(torch.float)
    boxes /= (bins - 1)
    return boxes
    



# prompt for task utils, to be moved
# theres a token weighting to make sure decoder doesnt predict the prompt token, obtain from build_prompt_seq_from_bbox

def build_prompt_seq_from_task_id(task_vocab_id: int,
                                 response_seq=None,
                                prompt_shape=None):
    """"Build prompt seq just using task id.
      Args:
          task_vocab_id: Vocab id for the task.
          response_seq: an (optional) discerte target sequen with shape (bsz, ..., k).
          prompt_shape: an (optional) tuple for prompt shape. One and only one of
          `response_seq` and `prompt_shape` should be specified.
      Returns:
          discrete input sequence of task id with shape (bsz, ..., 1).
            """
    task_id = torch.tensor(task_vocab_id).to(torch.int64)
    if response_seq is not None:
        prompt_seq = torch.zeros_like(response_seq[..., :1]) + task_id.to(response_seq.dtype)

    if prompt_shape is not None:
        assert response_seq is None, 'double specification'
        prompt_seq = torch.zeros(prompt_shape, dtype=torch.int64) + task_id.to(torch.int64)

    return prompt_seq




def pad_to_max_len(data, max_len, dim, padding_token=0):
  """Pad the data tensor to max length on dim."""
  shape = list(data.shape)
  padding_shape, new_shape = copy.copy(shape), copy.copy(shape)
  padding_shape[dim] = max_len - padding_shape[dim]
  new_shape[dim] = max_len
  paddings = torch.full(padding_shape, padding_token, dtype=data.dtype, device=data.device)
  return torch.cat([data, paddings], axis=dim).view(new_shape).contiguous()









def decode_object_seq_to_bbox(logits,
                              pred_seq,
                              quantization_bins,
                              coord_vocab_shift):
  """Decode objects (label & bbox) for seq from `build_response_seq_from_bbox`.

  Assume yxyxc format with truncation at the end for any uneven extra tokens.
    Replace class tokens with argmax instead of sampling.

  Args:
    logits: `float` output logits in shape of (bsz, max_seq_len, vocab_size).
    pred_seq: `int` pred sequence in shape of (bsz, max_seq_len).
    quantization_bins: `int` for bins.
    coord_vocab_shift: `int`, shifting coordinates by a specified integer.

  Returns:
    pred_class: `int` of shape (bsz, max_instances_per_image).
    pred_bbox: `float` of shape (bsz, max_instances_per_image, 4).
    pred_score: `float` of shape (bsz, max_instances_per_image).
  """
  DEFAULT_MAX = 500
  pred_seq = pred_seq[:,:DEFAULT_MAX]
  logits = logits[:,:DEFAULT_MAX]
  _, seqlen, vocab_size = logits.shape

  if seqlen % 5 != 0:  # truncate out the last few tokens.
    pred_seq = pred_seq[..., :-(seqlen % 5)]
    logits = logits[..., :-(seqlen % 5), :]


  
  logits_cp = logits.clone()
  logits_cp[:,:,(vocab.BASE_VOCAB_SHIFT + vocab.FAKE_CLASS_TOKEN)] = float(str('-inf'))

  pred_class_p = torch.softmax(logits_cp, -1)[:, 4::5]  # (bsz, instances, vocab_size)
  mask_s1 = [0.] * vocab.BASE_VOCAB_SHIFT  # reserved.
  mask_s2 = [1.] * (coord_vocab_shift - vocab.BASE_VOCAB_SHIFT)  # labels.
  mask_s3 = [0] * (vocab_size - coord_vocab_shift)  # coordinates and others.
  mask = torch.tensor(mask_s1 + mask_s2 + mask_s3)
  pred_class = torch.argmax(pred_class_p * mask[None, None, :], -1)


  ## replace the “noise” class label with a real class 
  ## label that has the highest likelihood among all real class labels
  real_pred_class_p = torch.softmax(logits, -1)[:, 4::5]
  pred_score = (real_pred_class_p * torch.nn.functional.one_hot(pred_class, vocab_size)).sum(-1)
  pred_class = torch.maximum(pred_class - vocab.BASE_VOCAB_SHIFT, torch.tensor(0))
  pred_bbox = seq_to_bbox(pred_seq - coord_vocab_shift, quantization_bins)
  

  return pred_class, pred_bbox, pred_score




def seq_to_bbox(seq, quantization_bins, seq_format='xyxy_name'):
  """Returns [0, 1] normalized xyxy bbox from token sequence.
     (Chun Hung) We follow the conventional way of normalized (x_min,y_min, x_max, y_max, c) , refer to data/bhx.py

  """
  # [batch, 5*num_instances]
  assert len(seq.shape) == 2, seq.shape
  # [batch, num_instances, 1]
  if seq_format.startswith('name'):
    xmin = seq[:, 1::5][...,None]
    ymin = seq[:, 2::5][...,None]
    xmax = seq[:, 3::5][...,None]
    ymax = seq[:, 4::5][...,None]
  else:
    xmin = seq[:, 0::5][...,None]
    ymin = seq[:, 1::5][...,None]
    xmax = seq[:, 2::5][...,None]
    ymax = seq[:, 3::5][...,None]

  if seq_format in ['name_cxcyhw', 'cxcyhw_name']:
    ycnt, xcnt, ysize, xsize = ymin, xmin, ymax, xmax
    ymin = ycnt - ysize//2
    xmin = xcnt - xsize//2
    ymax = ycnt + ysize//2
    xmax = xcnt + xsize//2
  quantized_box = torch.cat([xmin, ymin, xmax, ymax], axis=-1)
  quantized_box = dequantize(quantized_box, quantization_bins)
  return torch.minimum(torch.maximum(quantized_box, torch.tensor(0)), torch.tensor(1))




def decode_seq_to_polygon(
                          pred_seq,
                          quantization_bins,      
                          coord_vocab_shift):
    """
    Args:
    pred_seq: `int` pred sequence in shape of (bsz, max_seq_len).
    quantization_bins: `int` for bins.
    """
    bsz = pred_seq.shape[0]
    #if seqlen % 2 != 0:  # truncate out the last few tokens.
    #    pred_seq = pred_seq[..., :-(seqlen % 2)]
    #    logits = logits[..., :-(seqlen % 2), :]
    #pred_p = torch.softmax(logits, -1)
    sep_ind = (pred_seq == vocab.SEPARATOR_TOKEN)
    pred_seq = torch.maximum(pred_seq - coord_vocab_shift, torch.tensor(0))
    deq = dequantize(pred_seq, quantization_bins)
    deq = torch.minimum(deq, torch.tensor(1))
    
    post = []
    pred = []
    for b in range(bsz):
        l = []
        c = 0
        tup = sep_ind[b].nonzero()
        if tup.shape[0] > 0:
            for i in tup:
                seq = deq[b][c:i]
                if seq.shape[0] % 2 != 0:  # truncate out the last few tokens.
                    seq = seq[..., :-(seq.shape[0] % 2)]
                l += [seq.view(-1,1,2).cpu().numpy()]
                c = i + 1
                
            post += [l]
        else:
            seq = deq[b]
            if seq.shape[0] % 2 != 0:  # truncate out the last few tokens.
                seq = seq[..., :-(seq.shape[0] % 2)]
            l = [seq.view(-1,1,2).numpy()]
            post += [l]


    return post





class LossFunction():
    def __init__(self, loss_type=None):
        self.criterion = self.get_loss(loss_type)
        self.loss_type = loss_type

    def get_loss(self, loss_type : str = None):
        """
        Args:
            loss_type: `str` None then crossentropy, for ex focal@1 where 1 is parameter gamma.

        """
        if loss_type is None:
            return torch.nn.CrossEntropyLoss(reduction='none')


        elif 'focal' in loss_type:
            if '@' in loss_type:
                self.gamma = float(loss_type.split('@')[1])
            else:
                raise ValueError(f"Parameter not specified in {loss_type}, for ex focal@0.5")

            return torchvision.ops.focal_loss.sigmoid_focal_loss


        else:
            raise NotImplementedError(f"Loss {loss_type} not implemented.")

    def __call__(self, logits, target):
        """
        Args:
            logits: `floattensor` (bsz, c, seqlen) by default for cross entropy loss.
            target: `long tensor` (bsz, seqlen) by default for cross entropy loss
        """
        if self.loss_type == None:
            return self.criterion(logits, target)
        elif 'focal' in self.loss_type:
            target_onehot = torch.nn.functional.one_hot(target, logits.shape[-2]).float()
            p = torch.softmax(logits.permute(0,2,1),-1)
            logp = torch.log(p + 1e-8)
            focal_weight = torch.pow(1. - p, self.gamma) if self.gamma > 0 else 1.
            return focal_weight * target_onehot * logp

        else:
            raise NotImplementedError







def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        target = target.to(input.dtype)
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input_: Tensor, target: Tensor, reduce_batch_first: bool = False, reduce_label = True ,epsilon=1e-6, n_class=2):
    # Average of Dice coefficient for all classes
    assert input_.size() == target.size()
    if input_.ndim == 3:
        input_ = torch.nn.functional.one_hot(input_, num_classes=n_class).permute(0,-1,1,2)
        target = torch.nn.functional.one_hot(target, num_classes=n_class).permute(0,-1,1,2)
    if reduce_label:
        dice = 0
        for channel in range(input_.shape[1]):
            dice += dice_coeff(input_[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

        return dice / input_.shape[1]
    else:
        dice = np.zeros(input_.shape[1])
        for channel in range(input_.shape[1]):
            dice[channel] = dice_coeff(input_[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)
        return dice

