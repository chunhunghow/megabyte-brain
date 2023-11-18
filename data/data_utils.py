

import numpy as np
import torch
import random
import vocab

def truncation_bbox(bbox):
    return torch.minimum(torch.maximum(bbox, torch.tensor(0.)), torch.tensor(1.))

def shift_bbox(bbox, truncation=True):
    """
    Shifting bbox without changing the bbox height and width.
    Assume box is in xyxyn format. Chen Ting tf repo did yxyx.

    """
    n = bbox.shape[0]
    # randomly sample new bbox centers.
    cy = torch.rand(n, 1).to(bbox.device)
    cx = torch.rand(n, 1).to(bbox.device)
    w = bbox[:, 2:3] - bbox[:, 0:1]
    h = bbox[:, 3:4] - bbox[:, 1:2]
    bbox = torch.cat([cx - torch.abs(w)/2, cy - torch.abs(h)/2,
                      cx + torch.abs(w)/2, cy + torch.abs(h)/2], -1)
    return truncation_bbox(bbox) if truncation else bbox


def random_bbox(n, max_size=1.0, truncation=True):
    """
    Generating random n bbox with max size specified within [0, 1].
    Assume box is in xyxyn format. Chen Ting tf repo did yxyx.

    """
    # original imp from Chen Ting draw uniformly for cx cy, but our imp is for medical img
    # the box should be around center
    cx = torch.fmod(torch.randn(n,1),0.3)+0.5
    cy = torch.fmod(torch.randn(n,1),0.3)+0.5
    h = torch.fmod(torch.randn(n,1),0.2)+0.3 # will center around 30% of the image size 
    w = torch.fmod(torch.randn(n,1),0.2)+0.3
    bbox = torch.cat([cx - torch.abs(w)/2, cy - torch.abs(h)/2,
                      cx + torch.abs(w)/2, cy + torch.abs(h)/2], -1)
    return truncation_bbox(bbox) if truncation else bbox


def jitter_bbox(bbox, min_range=0., max_range=0.05, truncation=True):
    """
    Jitter the bbox.
    Assume box is in xyxyn format. Chen Ting tf repo did yxyx.

    Args:
      bbox: `float` tensor of shape (n, 4), ranged between 0 and 1.
      min_range: min jitter range in ratio to bbox size.
      max_range: max jitter range in ratio to bbox size.
      truncation: whether to truncate resulting bbox to remain [0, 1].

    Note:
      To create noisy positives, set min_range=0, which enables truncated normal
        distribution. max_range <=0.05: noisy duplicates, <=0.02: near duplicate.
      To create negatives: set min_range >= 0.1 to avoid false negatives;
        suggested max_range <=0.4 to avoid too much randomness.


    Returns:
      jittered bbox.
    """
    n = bbox.shape[0]
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    noise = torch.stack([w, h, w, h], -1)
    if min_range == 0:
      noise_rate = torch.fmod(torch.randn(n,4),max_range/5) 
    else:
      noise_rate1 = torch.rand(n, 4) * (max_range - min_range) + min_range
      noise_rate2 = -1*(torch.rand(n, 4) * (max_range - min_range) + min_range)
      selector = (torch.rand(n,4) < 0.5).to(torch.float)
      noise_rate = noise_rate1 * selector + noise_rate2 * (1. - selector)
    bbox = bbox + noise * noise_rate.to(bbox.device)
    return truncation_bbox(bbox) if truncation else bbox



def random_shuffle( x : torch.Tensor):
    idx = list(range(x.shape[0]))
    random.shuffle(idx)
    x = x[idx]
    return x, idx


def augment_bbox(bbox, bbox_label, max_jitter, n_noise_bbox, mix_rate=1.0):
    """Augment bbox.
   
    There are two types of noises to add:
      1. Bad bbox: jittered bbox, shifted bbox, or random bbox.
      2. Duplicated bbox.
   
    Args:
      bbox: `float` tensor of shape (n, 4), ranged between 0 and 1.
      bbox_label: `int` tensor of shape (n,).
      max_jitter: `float` scalar specifying max jitter range for positive bbox.
      n_noise_bbox: `int` scalar tensor specifying size of the extra noise to add.
      mix_rate: `float`. Probability of injecting the bad bbox in the middle of
        original bbox, followed by dup bbox at the end; otherwise simply append
        all noises at the end of original bbox.
        (Chun Hung) here i changed the definition of mix to whether to add noise(dup+bad)
        Best to keep at 1.
   
    Note:
      max_jitter should be smaller than 0.1 (for positives)

    Returns:
      bbox_new: augmented bbox that's `n_noise_bbox` larger than original.
      label_new: new label for bbox_new. Fake and Bad boxes are assigned FAKE_CLASS_TOKEN
      is_real: a `float` 0/1 indicator for whether a bbox is real.
      is_noise: a `float` 0/1 indicator for whether a bbox is extra.
    """
    n = bbox.shape[0]
    #dup_bbox_size = torch.randint(n_noise_bbox + 1, size=(1,))
    dup_bbox_size = n_noise_bbox - 5
    dup_bbox_size = 0 if n == 0 else dup_bbox_size
    bad_bbox_size = n_noise_bbox - dup_bbox_size
    multiplier = 1 if n == 0 else (n_noise_bbox//n + 1)
    bbox_tiled = bbox.repeat(multiplier, 1)

    # Create bad bbox.
    bbox_tiled, _ = random_shuffle(bbox_tiled)
    bad_bbox_shift = shift_bbox(bbox_tiled[:bad_bbox_size], truncation=True)
    bad_bbox_random = random_bbox(bad_bbox_size, max_size=1.0, truncation=True).to(bbox.device)
    bad_bbox = torch.cat([bad_bbox_shift, bad_bbox_random], 0)
    bad_bbox = random_shuffle(bad_bbox)[0][:bad_bbox_size]
    bad_bbox_label = torch.zeros([bad_bbox_size], dtype=bbox_label.dtype) + (
        vocab.FAKE_CLASS_TOKEN ) # originally FAKE_TOKEN - VOCAB_SHIFT

    # Create dup bbox.
    bbox_tiled, _ = random_shuffle(bbox_tiled)
    dup_bbox = jitter_bbox(
        bbox_tiled[:dup_bbox_size], min_range=0, max_range=2, truncation=True)
    dup_bbox_label = torch.zeros([dup_bbox_size], dtype=bbox_label.dtype) + (
        vocab.FAKE_CLASS_TOKEN) # can even have different token for dup, we shuffle noise label (now all 30)

    # Jitter positive bbox.
    if max_jitter > 0:
      bbox = jitter_bbox(bbox, min_range=0, max_range=max_jitter, truncation=True)

    
    if torch.rand(1) < mix_rate:
      # Append noise bbox to bbox and create mask.
      noise_bbox = torch.cat([bad_bbox, dup_bbox], 0)
      noise_bbox_label = torch.cat([bad_bbox_label, dup_bbox_label], 0)
      bbox_new = torch.cat([bbox, noise_bbox], 0)
      bbox_new_label = torch.cat([bbox_label, noise_bbox_label.to(bbox_label.device)], 0)  # with randomly associated class labels
      

    else:
      bbox_new = torch.cat([bbox, torch.zeros((n_noise_bbox, 4)).to(bbox.device)],0)
      bbox_new_label = torch.cat([bbox_label, torch.zeros([n_noise_bbox], dtype=bbox_label.dtype).to(bbox_label.device)]) # originally FAKE_TOKEN - VOCAB_SHIFT

    return bbox_new, bbox_new_label


## original
#def augment_bbox(bbox, bbox_label, max_jitter, n_noise_bbox, mix_rate=1.0):
#    """Augment bbox.
#   
#    There are two types of noises to add:
#      1. Bad bbox: jittered bbox, shifted bbox, or random bbox.
#      2. Duplicated bbox.
#   
#    Args:
#      bbox: `float` tensor of shape (n, 4), ranged between 0 and 1.
#      bbox_label: `int` tensor of shape (n,).
#      max_jitter: `float` scalar specifying max jitter range for positive bbox.
#      n_noise_bbox: `int` scalar tensor specifying size of the extra noise to add.
#      mix_rate: `float`. Probability of injecting the bad bbox in the middle of
#        original bbox, followed by dup bbox at the end; otherwise simply append
#        all noises at the end of original bbox.
#        (Chun Hung) here i changed the definition of mix to whether to add noise(dup+bad)
#        Best to keep at 1.
#   
#    Note:
#      max_jitter should be smaller than 0.1 (for positives)
#
#    Returns:
#      bbox_new: augmented bbox that's `n_noise_bbox` larger than original.
#      label_new: new label for bbox_new. Fake and Bad boxes are assigned FAKE_CLASS_TOKEN
#      is_real: a `float` 0/1 indicator for whether a bbox is real.
#      is_noise: a `float` 0/1 indicator for whether a bbox is extra.
#    """
#    n = bbox.shape[0]
#    dup_bbox_size = torch.randint(n_noise_bbox + 1, size=(1,))
#    dup_bbox_size = 0 if n == 0 else dup_bbox_size
#    bad_bbox_size = n_noise_bbox - dup_bbox_size
#    multiplier = 1 if n == 0 else (n_noise_bbox//n + 1)
#    bbox_tiled = bbox.repeat(multiplier, 1)
#
#    # Create bad bbox.
#    bbox_tiled, _ = random_shuffle(bbox_tiled)
#    bad_bbox_shift = shift_bbox(bbox_tiled[:bad_bbox_size], truncation=True)
#    bad_bbox_random = random_bbox(bad_bbox_size, max_size=1.0, truncation=True).to(bbox.device)
#    bad_bbox = torch.cat([bad_bbox_shift, bad_bbox_random], 0)
#    bad_bbox = random_shuffle(bad_bbox)[0][:bad_bbox_size]
#    bad_bbox_label = torch.zeros([bad_bbox_size], dtype=bbox_label.dtype) + (
#        vocab.FAKE_CLASS_TOKEN ) # originally FAKE_TOKEN - VOCAB_SHIFT
#
#    # Create dup bbox.
#    bbox_tiled, _ = random_shuffle(bbox_tiled)
#    dup_bbox = jitter_bbox(
#        bbox_tiled[:dup_bbox_size], min_range=0, max_range=1, truncation=True)
#    dup_bbox_label = torch.zeros([dup_bbox_size], dtype=bbox_label.dtype) + (
#        vocab.FAKE_CLASS_TOKEN) # can even have different token for dup, we shuffle noise label (now all 30)
#
#    # Jitter positive bbox.
#    if max_jitter > 0:
#      bbox = jitter_bbox(bbox, min_range=0, max_range=max_jitter, truncation=True)
#
#    if torch.rand(1) < mix_rate:
#      # Mix the bbox with bad bbox, appneded by dup bbox.
#      bbox_new = torch.cat([bbox, bad_bbox], 0)
#      bbox_new_label = torch.cat([bbox_label, bad_bbox_label], 0)
#      idx = list(range(box_new.shape[0]))
#      random.shuffle(idx)
#      bbox_new = torch.gather(bbox_new, idx)
#      bbox_new_label = torch.gather(bbox_new_label, idx)
#      bbox_new = torch.cat([bbox_new, dup_bbox], 0)
#      bbox_new_label = torch.cat([bbox_new_label, dup_bbox_label], 0)
#    else:
#      # Merge bad bbox and dup bbox into noise bbox.
#      noise_bbox = torch.cat([bad_bbox, dup_bbox], 0)
#      noise_bbox_label = torch.cat([bad_bbox_label, dup_bbox_label], 0)
#
#    
#    bbox_new = torch.cat([bbox, torch.zeros((n_noise_bbox, 4)).to(bbox.device)],0)
#    bbox_new_label = torch.cat([bbox_label, torch.zeros([n_noise_bbox], dtype=bbox_label.dtype).to(bbox_label.device)]) # originally FAKE_TOKEN - VOCAB_SHIFT
#
#    return bbox_new, bbox_new_label
