## Token 100-1000 is for classification, some is used for detection label


from typing import Any, Dict, List
import torch
from tasks import task as task_lib
import util
import vocab
import random
from torchmetrics import ConfusionMatrix
import numpy as np
from einops import rearrange

@task_lib.TaskRegistry.register('classification')
class TaskClassification(task_lib.Task):

    def __init__(self, config):
        super().__init__(config)
        self.reset()
 

    def reset(self):
        self.meter = 0.


    def compute(self):
        """
        Assume self.meter is tensor size (C,2,2), where C is the number of classes
        Torchmetrics confusionMatrix has target on vertical axis, and prediction on horizontal.

        Return:
              Precision : [Cx1]
        """
        #return ( self.meter[:,1,1] ) / self.meter.sum((1,2))
        return  self.meter[:,1,1]  / (self.meter[:,1,1] + self.meter[:,0,1] + 1e-6)



    def preprocess_target(self, batch_targets, idx=None):

        """
        This task will receive classification data, the target must be one hot (multiclass or multilabel),
        usually its multilabel so we fix multilabel as default for now.

        Args:
            batch_targets : List[Tensor], first index is the input image batch, second is the target in (n,C)
        Return:
        """
        self.criterion = ConfusionMatrix(task='multilabel', num_labels=batch_targets[1].shape[1] )
        im = batch_targets[0]
        b = im.shape[0]
        im = rearrange(im, "b c (h p1) (w p2) -> b (c h w) (p1 p2)", p1=self.cfg['patch_size_sqrt'], p2=self.cfg['patch_size_sqrt'])
        im = im.view(b, -1)

        return {'image': im ,'label':batch_targets[1]}




    def preprocess_batched(self, batched_examples ,training=True, idx=None):
        """
       Typical operations in this preprocessing step for detection task:
       - Quantization and serialization of object instances.
       - Creating the input sequence, target sequence, and token weights.

       Args:
           batched_examples: tuples of feature and label tensors that are
                     preprocessed, batched, and stored with `dict`. FasteRCNN, DINO style.
           training: bool.

       Returns:
           images: `float` of shape (bsz, h, w, c)
           input_seq: `int` of shape (bsz, seqlen).
           target_seq: `int` of shape (bsz, seqlen).
           token_weights: `float` of shape (bsz, seqlen)

        """
        config = self.cfg
        ret = build_response_seq_from_label(
            batched_examples['label'],
            config.cls_vocab_shift, self.cfg.max_seq_len)
        response_seq, response_seq_cm, token_weights = ret
        prompt_seq = util.build_prompt_seq_from_task_id(
            self.task_vocab_id, response_seq)  # (bsz, 1)

        # prompt token
        input_seq = torch.cat([prompt_seq, response_seq_cm], -1) 
        target_seq = torch.cat([prompt_seq, response_seq], -1)
        # Pad sequence to a unified maximum length.
        #input_seq = util.pad_to_max_len(response_seq_cm, config.max_seq_len + 1, -1) #pad one for BOS
        #target_seq = util.pad_to_max_len(response_seq, config.max_seq_len + 1, -1, padding_token= vocab.PADDING_TOKEN) # pad one for EOS

        #target seq will not have prompt token
        input_seq, target_seq = input_seq[..., :-1], target_seq[..., 1:]
        token_weights = util.pad_to_max_len(token_weights, config.max_seq_len , -1)

        # for MEGA, we want to try directly predict from encoder, instead of giving in decoder
        input_seq[:,1:] = torch.zeros_like(input_seq[:,1:])

        # Assign lower weights for ending/padding tokens. padding assign eos_token_weight = 0.1
        # EOS token is same as padding token. Same weight.
        # WIth SeqAug, there will not be any padding token 
        #token_weights = torch.where(
        #    target_seq == vocab.PADDING_TOKEN,
        #    torch.zeros_like(token_weights) + config.eos_token_weight, token_weights)

        if training:
          return batched_examples['image'], input_seq, target_seq, token_weights
        else:
          return batched_examples['image'], response_seq, batched_examples


    def postprocess(self,im, target, logits, dataset_obj, fname='images_train.jpg', save=True, idx=None):
        """
        Decode to labels

        target : List[Tensor] , binary label
        save : For storing images, not used in classification task.
        dataloader_idx: Ununsed, only for coding purpose.
        """
        assert hasattr(dataset_obj , 'TARGET_LABELS')
        if logits.ndim >= 2:
            pred_seq = torch.argmax(torch.softmax(logits.permute(0,2,1),-1),-1).cpu() #logits -> [bsz seqlen token]
        else:
            pred_seq = logits
        separation = (pred_seq ==vocab.SEPARATOR_TOKEN).nonzero()
        mask = torch.ones_like(pred_seq)
        for tup in separation: #not all rows predicted separation
            mask[tup[0]][tup[1]:] = 0 
        mask = mask.to(pred_seq.device)
        shifted = torch.where((pred_seq > (len(dataset_obj.TARGET_LABELS)+self.cfg.cls_vocab_shift-1)) | (pred_seq < self.cfg.cls_vocab_shift), torch.zeros_like(pred_seq), pred_seq)
        shifted = shifted * mask
        #shifted = torch.where((shifted != 0) , shifted - self.cfg.cls_vocab_shift, shifted ) #200 will be zero
        post = []
        for b in range(shifted.shape[0]):
            if len(shifted[b][shifted[b] > 0]) == 0:
                post += [torch.zeros(1, target.shape[1])]
            else:
                #out = list(set(np.array(dataset_obj.TARGET_LABELS)[shifted[b][shifted[b] >0] - self.cfg.cls_vocab_shift]))
                out = (shifted[b][shifted[b] >0] - self.cfg.cls_vocab_shift).unique()
                post += [torch.nn.functional.one_hot(out, num_classes=target.shape[-1]).sum(0, keepdims=True)]

        
        self.meter += self.criterion(torch.cat(post), target.cpu())
        return post



    def log_result(self, result,dataset_obj, mode='train'):
        name = [f"{mode}_Precision_{i}" for i in range(len(result))]
        return dict(zip(name, result))


def build_response_seq_from_label(label, token_shift, seqlen):
    """  
    Args:
        label: `Tensor` (n, C), binary
        token_shift: `int` The base number to add for the range of label tokens
#    Returns:
#      discrete sequences with shape (bsz, seqlen).
    """



    #==================================================
    #    Shift tokens to designed range
    #==================================================
    _C = label.shape[1]
    vec = torch.arange(token_shift, token_shift + _C)[None,].to(label.device)
    target_seq = label * vec
    target_seq = torch.cat([target_seq, torch.zeros(label.shape[0])[...,None].to(target_seq.device) + vocab.SEPARATOR_TOKEN],1)
    target_seq = target_seq.sort(descending=True).values

    separation = (target_seq ==vocab.SEPARATOR_TOKEN).nonzero()
    for tup in separation: #not all rows predicted separation
        target_seq[tup[0], :tup[1]] = target_seq[tup[0], :tup[1]][torch.randperm(tup[1])]

    response = torch.zeros(target_seq.shape[0], seqlen).to(label.device).long()
    response[:, :target_seq.shape[1]] = target_seq
    token_weights = (response != 0).to(torch.float)

    return response, response, token_weights




