


from typing import Any, Dict, List
import torch
from tasks import task as task_lib
import util
import vocab
import random
from einops import rearrange
from data.data_utils import augment_bbox
from vis_utils import plot_images_with_boxes
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import wandb
import numpy as np
from util import box_cxcywh_to_xyxy

@task_lib.TaskRegistry.register('object_detection')
class TaskObjectDetection(task_lib.Task):

    def __init__(self, config):
        super().__init__(config)
        self.reset()
 
    def reset(self):
        self.criterion = MeanAveragePrecision(class_metrics=True)

    def compute(self):
        torchmetrics_map = self.criterion.compute()
        #torchmetrics_map = dict([(f'{mode}_'+k, v) for k,v in torchmetrics_map.items()])
        return torchmetrics_map

    def preprocess_target(self, out, object_order='random', idx=None):
        """
        Object ordering must be called _before_ padding to max instances.
        Refer to data_utils.py preprocess_train()

        Args:
            out : (im, batch_targets)
                batch_targets : List[Dict], dictionary containing key 'boxes' and 'labels'
        Return:
            Dict with box [bsz n 4] , label [bsz n] in tensor.
        """
        batch_targets = out[1]
        assert isinstance(batch_targets, list) , 'Expect a list of dictionary for detection targets. {"boxes": [[..]], "labels":[...]}'
        try:
            max_target = max([len(k['labels']) for k in batch_targets])
        except KeyError:
            raise Exception('Expect a list of dictionary for detection targets. {"boxes": [[..]], "labels":[...]}')

        bsz = len(batch_targets)
        batched_box = []
        batched_label = []

        for d in batch_targets:
            box = d['boxes']
            label = d['labels']
            assert (len(box.shape) == 2) or (box.shape[1] == 4), 'Bbox should have dimension [N 4]'

            #========================================
            #          Object ordering
            #=========================================
            if object_order == 'random':
                if box.shape[0] == 1:
                    pass
                else:
                    ind = list(range(box.shape[0]))
                    random.shuffle(ind)
                    box = box[ind]
                    label = label[ind]
            else:
                raise NotImplementedError

            #=========================================
            #           Sequence Augmentation
            #========================================
            box_new, label_new = augment_bbox(box, label, 0.2, self.cfg.max_instances_per_image_test - box.shape[0])
 
            batched_box += [box_new[None,]]
            batched_label += [label_new[None,]]

        #for megabyte
        im = out[0]
        b = im.shape[0]
        im = rearrange(im, "b c (h p1) (w p2) -> b (c h w) (p1 p2)", p1=self.cfg['patch_size_sqrt'], p2=self.cfg['patch_size_sqrt'])
        im = im.view(b, -1)

        return {'image' : im , 'bbox': torch.cat(batched_box), 'label': torch.cat(batched_label)}
                







    def preprocess_batched(self, batched_examples, training=True, idx=None):
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
        config = self.cfg #TODO changed to just self.cfg (use task instead of model)
        ret = build_response_seq_from_bbox(
            batched_examples['bbox'], batched_examples['label'],
            config.quantization_bins, config.noise_bbox_weight,
            config.coord_vocab_shift,
            class_label_corruption=config.class_label_corruption)
        response_seq, response_seq_cm, token_weights = ret
        prompt_seq = util.build_prompt_seq_from_task_id(
            self.task_vocab_id, response_seq)  # (bsz, 1)

        #===========================================================
        # Since we are using MegaByte, the local model only restricted to 
        # 64 tokens per patch, we want all information to be in the first patch
        # so it wouldnt be affected by global model during inference.
        # During inference, we will sample the token each time step.
        #===========================================================
        response_seq[:,61:] = 0 
        response_seq_cm[:,61:] = 0 
        token_weights[:,61:] = 0
        #===========================================================


        # prompt token
        input_seq = torch.cat([prompt_seq, response_seq_cm], -1) 
        target_seq = torch.cat([prompt_seq, response_seq], -1)
        # Pad sequence to a unified maximum length.
        #input_seq = util.pad_to_max_len(response_seq_cm, config.max_seq_len + 1, -1) #pad one for BOS
        #target_seq = util.pad_to_max_len(response_seq, config.max_seq_len + 1, -1, padding_token= vocab.PADDING_TOKEN) # pad one for EOS

        #target seq will not have prompt token
        input_seq, target_seq = input_seq[..., :-1], target_seq[..., 1:]
        input_seq = util.pad_to_max_len(input_seq, config.max_seq_len , -1)
        target_seq = util.pad_to_max_len(target_seq, config.max_seq_len , -1)
        token_weights = util.pad_to_max_len(token_weights, config.max_seq_len , -1)

        # Assign lower weights for ending/padding tokens. padding assign eos_token_weight = 0.1
        # EOS token is same as padding token. Same weight.
        # WIth SeqAug, there will not be any padding token 
        token_weights = torch.where(
            target_seq == vocab.PADDING_TOKEN,
            torch.zeros_like(token_weights) + config.eos_token_weight, token_weights)


        if training:
          return batched_examples['image'], input_seq, target_seq, token_weights
        else:
          return batched_examples['image'], response_seq, batched_examples




    def postprocess(self,im,target, logits, dataset_obj, fname='images_train.jpg', save=True, idx=None):
        """
        Args:
            save: For storing images.
            dataloader_idx: Ununsed, for coding purpose.
        """
        img_size = dataset_obj.cfg.img_size
        b = logits.shape[0]

        #=============================
        # MEGABYTE
        #=============================
        if len(im.shape) == 2:
            im = rearrange(im, "b (k p) -> b k p", p=self.cfg['patch_size_sqrt']**2)
            im = rearrange(im, "b (h w) (p1 p2) -> b (h p1) (w p2)", h = np.sqrt(im.shape[1]).astype(int), p1=self.cfg['patch_size_sqrt'], p2=self.cfg['patch_size_sqrt'])
            im = im.view(b,1, img_size, img_size)
        #=============================
        #=============================
        pred_seq = torch.argmax(torch.softmax(logits.permute(0,2,1),-1),-1).cpu() #logits -> [bsz seqlen token]
        pred_cls, pred_box, pred_score = \
          util.decode_object_seq_to_bbox(logits.permute(0,2,1).cpu(), pred_seq, self.cfg.quantization_bins, self.cfg.coord_vocab_shift)
        pred_box = pred_box * img_size #plot_images take xyxy for pred box
        post = self.make_targets(pred_box, pred_cls, pred_score, 'cuda')
        if save:
            plot_images_with_boxes(im , target , post, names= dataset_obj.cfg.cls_names, fname= fname, target_format='xyxyn', save=save) 

        # target xyxyn -> xyxy for torchmetric mAP
        H, W = im.shape[-2:]
        for i in range(im.shape[0]):
            box = target[i]['boxes']
            box[:,0] *= W
            box[:,1] *= H
            box[:,2] *= W
            box[:,3] *= H
            target[i]['boxes'] = box

        self.criterion.update(post,target)
        return post


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


    def log_result(self, torchmetrics_map, dataset_obj, mode='val'):
        """
        Return:
            `wandb Table`
        """

        ## torchmetrics
        map_per_class = torchmetrics_map['map_per_class']
        columns = ['Class', 'mAP']
        table = wandb.Table(columns=columns)

        table.add_data(*[ 'All', torchmetrics_map['map']])
        for i,c in enumerate(map_per_class):
            data = [dataset_obj.cfg.cls_names[i], c ]
            table.add_data(*data)
        #self.logger.log_metrics({f'{mode} Table (Torchmetrics)' : table})
        #del torchmetrics_map['map_per_class']
        #del torchmetrics_map['mar_100_per_class']

        #torchmetrics_map = dict([(f'{mode}_'+k, v) for k,v in torchmetrics_map.items()])
        return table



def build_response_seq_from_bbox(bbox,
                                 label,
                                 quantization_bins,
                                 noise_bbox_weight,
                                 coord_vocab_shift,
                                 class_label_corruption='rand_cls'):
    """"Build target seq from bounding bboxes for object detection.
    In paper, Chen Ting stated that ... The target sequence y˜ in conventional autoregressive language
    modeling (i.e., with no sequence augmentation) is the same as the input sequence y....

    So modify target sequences so that the model can learn to identify the noise tokens rather than
    mimic them. Check ar_model.py compute_loss().

    (Chun Hung) My insight on seq aug is to make model learn to generate more boxes but also more precise in real boxes
    by learning to classify noise box. Also it will learn generate confident boxes in front. In another case when we simply
    pad 0 as EOS, theres a local minima where once model return 0 for one token, the following seq will just be attend to 0.
    Also it is brilliant to assign 0 weight for fake coordinate so the model will learn to generate box as realistic as possible.


    Objects are serialized using the format of xyxyc.

    Args:
      box: `float` bounding box of shape (bsz, n, 4). May contain dup. and bad boxes.
      label: `int` label of shape (bsz, n). If corresponding box is augmented, label is FAKE_CLASS_TOKEN
      quantization_bins: `int`.
      noise_bbox_weight: `float` on the token weights for noise bboxes.
      coord_vocab_shift: `int`, shifting coordinates by a specified integer.
      class_label_corruption: `string` specifying how labels are corrupted for the
        input_seq.

    Returns:
      discrete sequences with shape (bsz, seqlen).
    """

    #==================================================
    #    Shift tokens to designed range, padding remain
    #==================================================
    is_padding = (bbox.sum(-1) == 0)[...,None] ## if box = 0,0,0,0 then its a padding
    quantized_bbox = util.quantize(bbox, quantization_bins)
    quantized_bbox = quantized_bbox + coord_vocab_shift
    quantized_bbox = torch.where(is_padding, torch.zeros_like(quantized_bbox), quantized_bbox)
    is_fake = (label == vocab.FAKE_CLASS_TOKEN)[...,None]
    target_label = (label + vocab.BASE_VOCAB_SHIFT)[...,None] #shift 
    target_label = torch.where(is_padding, torch.zeros_like(target_label), target_label) #unshift the padding
    #================================================================================================================
    #    Target box should assign fake token to fake box
    #    Model will learn to classify class of noise box as noise class (FAKE_CLASS)
    #================================================================================================================
    target_quantized_bbox = torch.where(is_fake, torch.zeros_like(quantized_bbox)+vocab.FAKE_CLASS_TOKEN ,quantized_bbox )
    # get unique classes
    uniq = target_label.unique()
    uniq = uniq[(uniq != vocab.PADDING_TOKEN) & (uniq != (vocab.FAKE_CLASS_TOKEN + vocab.BASE_VOCAB_SHIFT))]

    input_label = torch.where(is_fake, uniq[torch.randint(len(uniq), size=target_label.shape)], target_label)


    response_seq = torch.cat([target_quantized_bbox, target_label], axis=-1)
    response_seq = response_seq.flatten(-2)

    response_seq_m = torch.cat([quantized_bbox, input_label], axis=-1) #input class is randomly associated
    response_seq_m = response_seq_m.flatten(-2)

    is_fake = is_fake.float()
    bbox_weight = torch.tile(1 - is_fake, [1, 1, 4])
    label_weight = (1- is_fake) + (is_fake) * noise_bbox_weight
    token_weights = torch.cat([bbox_weight, label_weight], -1) #label weight is 1 for all, only noise box should not be mimic
    token_weights = token_weights.flatten(-2)


    return response_seq, response_seq_m, token_weights





## original 
#def build_response_seq_from_bbox(bbox,
#                                 label,
#                                 quantization_bins,
#                                 noise_bbox_weight,
#                                 coord_vocab_shift,
#                                 class_label_corruption='rand_cls'):
#    """"Build target seq from bounding bboxes for object detection.
#    In paper, Chen Ting stated that ... The target sequence y˜ in conventional autoregressive language
#    modeling (i.e., with no sequence augmentation) is the same as the input sequence y....
#
#    So modify target sequences so that the model can learn to identify the noise tokens rather than
#    mimic them. Check ar_model.py compute_loss().
#
#    Objects are serialized using the format of yxyxc.
#
#    Args:
#      box: `float` bounding box of shape (bsz, n, 4).
#      label: `int` label of shape (bsz, n).
#      quantization_bins: `int`.
#      noise_bbox_weight: `float` on the token weights for noise bboxes.
#      coord_vocab_shift: `int`, shifting coordinates by a specified integer.
#      class_label_corruption: `string` specifying how labels are corrupted for the
#        input_seq.
#
#    Returns:
#      discrete sequences with shape (bsz, seqlen).
#    """
#    is_padding = (label == 0)[...,None]
#    quantized_bbox = util.quantize(bbox, quantization_bins)
#    quantized_bbox = quantized_bbox + coord_vocab_shift
#    quantized_bbox = torch.where(is_padding, torch.zeros_like(quantized_bbox), quantized_bbox)
#    new_label = (label + vocab.BASE_VOCAB_SHIFT)[...,None]
#    new_label = torch.where(is_padding, torch.zeros_like(new_label), new_label)
#
#    lb_shape = new_label.shape
#    # Bbox and label serialization.
#    response_seq = torch.cat([quantized_bbox, new_label], axis=-1)
#    response_seq = response_seq.flatten(-2)
#    rand_cls = torch.randint_like(new_label, vocab.BASE_VOCAB_SHIFT, coord_vocab_shift )
#    fake_cls = vocab.FAKE_CLASS_TOKEN + torch.zeros_like(new_label)
#    rand_n_fake_cls = torch.where(torch.rand(lb_shape).to(new_label.device)>0.5, rand_cls, fake_cls )
#    real_n_fake_cls = torch.where(torch.rand(lb_shape).to(new_label.device)>0.5, fake_cls, new_label )
#    real_n_rand_n_fake_cls = torch.where(torch.rand(lb_shape).to(new_label.device)>0.5, rand_n_fake_cls, new_label )
#    label_mapping = {'none': new_label,
#                      'rand_cls': rand_cls,
#                      'real_n_fake_cls': real_n_fake_cls,
#                      'rand_n_fake_cls': rand_n_fake_cls,
#                      'real_n_rand_n_fake_cls': real_n_rand_n_fake_cls}
#    new_label_m = label_mapping[class_label_corruption]
#    new_label_m = torch.where(is_padding, torch.zeros_like(new_label_m), new_label_m)
#    response_seq_m = torch.cat([quantized_bbox, new_label_m], axis=-1)
#    response_seq_m = response_seq_m.flatten(-2)
#
#    is_real = (new_label != vocab.BASE_VOCAB_SHIFT).to(float)
#    bbox_weight = torch.tile(is_real, [1, 1, 4])
#    label_weight = is_real + (1. - is_real) * noise_bbox_weight
#    token_weights = torch.cat([bbox_weight, label_weight], -1)
#    token_weights = token_weights.flatten(-2)
#
#    return response_seq, response_seq_m, token_weights
#
#
