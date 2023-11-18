import cv2
import numpy as np
import torch
from torchvision.utils import draw_segmentation_masks, make_grid
from PIL import Image
import wandb

#_color_getter = color_sys(100)

# plot known and unknown box
def add_box_to_img(img, boxes, colorlist, brands=None, in_format = 'cxcywhn'):
    """[summary]

    Args:
        img ([type]): np.array, H,W,3
        boxes ([type]): list of list(4)
        colorlist: list of colors.
        brands: text.

    Return:
        img: np.array. H,W,3.
    """
    assert in_format in ('cxcywhn', 'xywhn', 'xyxyn', 'xyxy', 'cxcywh')
    H, W = img.shape[:2]
    for _i, (box, color) in enumerate(zip(boxes, colorlist)):
        if in_format == 'cxcywhn':
            x, y, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H
        elif in_format == 'xywh':
            w, h = box[2], box[3]
            x, y = box[0] + w/2 , box[1] + h/2 
        elif in_format == 'xyxy':
            w = (box[2] - box[0])
            h = (box[3] - box[1])
            x, y = box[0]+ w/2, box[1] + h/2 #cx ,cy
        elif in_format == 'xyxyn':
            w = (box[2] - box[0]) *W
            h = (box[3] - box[1]) *H
            x, y = W* box[0]+ w/2, H* box[1] + h/2 #cx ,cy
        else:
            raise NotImplementedError('Box format only allow "cxcywhn", "xywh", "xyxy", "xyxyn" for now.')
        img = cv2.rectangle(img.copy(), (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color, 2)
        if brands is not None:
            brand = brands[_i]
            org = (int(x-w/2), int(y+h/2))
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            thickness = 1
            img = cv2.putText(img.copy(), str(brand), org, font, 
                fontScale, color, thickness, cv2.LINE_AA)
    return img

def plot_dual_img(img, boxes, labels, idxs, probs=None):
    """[summary]

    Args:
        img ([type]): 3,H,W. tensor.
        boxes (): tensor(Kx4) or list of tensor(1x4).
        labels ([type]): list of ints.
        idxs ([type]): list of ints.
        probs (optional): listof floats.

    Returns:
        img_classcolor: np.array. H,W,3. img with class-wise label.
        img_seqcolor: np.array. H,W,3. img with seq-wise label.
    """

    boxes = [i.cpu().tolist() for i in boxes]
    img = (renorm(img.cpu()).permute(1,2,0).numpy() * 255).astype(np.uint8)
    # plot with class
    class_colors = [_color_getter(i) for i in labels]
    if probs is not None:
        brands = ["{},{:.2f}".format(j,k) for j,k in zip(labels, probs)]
    else:
        brands = labels
    img_classcolor = add_box_to_img(img, boxes, class_colors, brands=brands)
    # plot with seq
    seq_colors = [_color_getter((i * 11) % 100) for i in idxs]
    img_seqcolor = add_box_to_img(img, boxes, seq_colors, brands=idxs)
    return img_classcolor, img_seqcolor


def plot_raw_img(img, boxes, labels):
    """[summary]

    Args:
        img ([type]): 3,H,W. tensor. 
        boxes ([type]): Kx4. tensor
        labels ([type]): K. tensor.

    return:
        img: np.array. H,W,3. img with bbox annos.
    
    """
    img = (renorm(img.cpu()).permute(1,2,0).numpy() * 255).astype(np.uint8)
    H, W = img.shape[:2]
    for box, label in zip(boxes.tolist(), labels.tolist()):
        x, y, w, h = box[0] * W, box[1] * H, box[2] * W, box[3] * H

        img = cv2.rectangle(img.copy(), (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), _color_getter(label), 2)
        # add text
        org = (int(x-w/2), int(y+h/2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 1
        img = cv2.putText(img.copy(), str(label), org, font, 
            fontScale, _color_getter(label), thickness, cv2.LINE_AA)

        return img



def plot_images_with_boxes(images, targets, preds=None, paths=None, fname='images.jpg', names=None, color=(0,255,0), conf_thres=0.25, target_format='cxcywhn', save=True):
    '''
    For detection task only.
    Received numpy or tensor.
    Args:

    targets : List[Dict] , contain 'labels' and 'boxes', if targets are prediction , must remain same structure
    preds: List[Dict], has "scores"
    names : class namea
    color : Color in BGR
    '''
    # Plot image grid with labels
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    #if isinstance(targets, torch.Tensor):
    #    targets = targets.cpu().numpy()


    max_size = 1920  # max image size
    max_subplots = 16  # max image subplots, i.e. 4x4
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
             break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        coord = targets[i]['boxes']
        cls = targets[i]['labels']


        if isinstance(coord, torch.Tensor):
             coord = coord.cpu().numpy()
             cls = cls.cpu().numpy()

        assert cls.dtype == int, 'Output label must be integer'

        txt = [names[c] for c in cls] if names is not None else cls
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        im = add_box_to_img( im , coord, [color]*len(coord), brands=txt, in_format=target_format ) #make sure im is numpy

        if preds is not None:
            strong =  torch.where(preds[i]['scores'] > conf_thres)[0] 
            coord = preds[i]['boxes'][strong]
            cls = preds[i]['labels'][strong]
            try:
                txt = [names[c] for c in cls] if names is not None else cls
            except IndexError as e:
                txt = cls # during early training, its possible that it returns out of range classes

            scores = preds[i]['scores'][strong]
            txt = [f'{t} {round(scores[i].item(), 3)} ' for i,t in enumerate(txt)]
            im = add_box_to_img( im , coord, [(255,0,0)]*len(coord), brands=txt, in_format= 'xyxy' ) # coord is in xyxy for Postprocess output

        mosaic[y:y + h, x:x + w, :] = im
    if fname[-4:] not in ('.jpg', '.png'):
        fname += '.png'
    if save:
        Image.fromarray(mosaic).save(fname)
    #cv2.imwrite(fname, mosaic)


def plot_aug_images(images, real_targets, fake_targets, paths=None, fname='images.jpg', names=None, color=(0,255,0), conf_thres=0.25, target_format='cxcywhn'):
    '''
    Received numpy or tensor.
    targets : List[Dict] , contain labels and bbox, if targets are prediction , must remain same structure
    preds: List[Dict], has "scores"
    names : class namea
    color : Color in BGR
    '''
    # Plot image grid with labels
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    #if isinstance(targets, torch.Tensor):
    #    targets = targets.cpu().numpy()


    max_size = 1920  # max image size
    max_subplots = 16  # max image subplots, i.e. 4x4
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
             break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        coord = real_targets[i]['boxes']
        cls = real_targets[i]['labels']


        if isinstance(coord, torch.Tensor):
             coord = coord.cpu().numpy()
             cls = cls.cpu().numpy()

        assert cls.dtype == int, 'Output label must be integer'
        txt = [names[c] for c in cls] if names is not None else cls

        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        im = add_box_to_img( im , coord, [(0,255,0)]*len(coord), brands=txt, in_format=target_format ) #make sure im is numpy


        #fake targets

        coord = fake_targets[i]['boxes']
        cls = fake_targets[i]['labels']


        if isinstance(coord, torch.Tensor):
             coord = coord.cpu().numpy()
             cls = cls.cpu().numpy()

        assert cls.dtype == int, 'Output label must be integer'
        txt = [names[c] for c in cls] if names is not None else cls

        im = add_box_to_img( im , coord, [(0,0,255)]*len(coord), brands=txt, in_format=target_format ) #make sure im is numpy
        mosaic[y:y + h, x:x + w, :] = im


    if fname[-4:] not in ('.jpg', '.png'):
        fname += '.png'
    
    Image.fromarray(mosaic).save(fname)
    #cv2.imwrite(fname, mosaic)



def plot_polygon(im, polygons,color=(0,0,255) ):
    """
    Args:
        im: `Tensor` (c,h,w)
        polygons: `List[Tensor]` List of (n_points, 1, 2) , padding removed.
        
    """
    if isinstance(im, torch.Tensor):
        im = im.cpu().float().numpy()
    if im.shape[-1] != 3:
       im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    #im = add_box_to_img( images , boxes, [(0,255,0)]*len(boxes), brands=None, in_format='xyxy' ) #make sure im is numpy
    if im.dtype != np.uint8:
        im = (im * 255).astype(np.uint8)
    for polygon in polygons:
        polygon = (polygon*im.shape[0]).astype(np.int32)
        #for pt in polygon: 
        #    cv2.circle(im, (pt[0][0], pt[0][1]), 3, (0,255,0), -1)
        #im = cv2.drawContours(im, polygon, -1, line_color, -1, cv2.LINE_AA)
        #im = cv2.drawContours(im, [polygon], -1, color, -1)
        if len(polygon) == 0:
            mask = np.zeros(im.shape[:2])
        else:
            mask = cv2.drawContours(np.zeros_like(im), [polygon], -1, (1,1,1), -1 )[:,:,0]

    return  mask


def plot_images_with_polygon(images, polygons, target, fname, target_format, save=True):
    """
    Args:
        images: `Tensor` (bsz,c,h,w), 0-1
        polygons: `List[List[Tensor]]` The list contains bsz of list, each inner list containing (n_points, 1, 2) , padding removed.
        target: `tensor` (bs,H,W) binary array
        boxes : `Tensor` (bsz,4)
        target_format : `str` polygons and boxes format. xyxy 
    """
    #max_size = 1920  # max image size
    #max_subplots = 16  # max image subplots, i.e. 4x4
    bs, _, h, w = images.shape  # batch size, _, height, width
    #bs_plot = min(bs, max_subplots)  # limit plot images
    #ns = np.ceil(bs ** 0.5)  # number of subplots (square)
    #mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    images = images.cpu()
    post = []
    grid = []
    #for i in range(bs):
    #    x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
    #    im = images[i].permute(1, 2, 0)
    #    pred_mask = plot_polygon(im, polygons[i], color=(255,0,0))
    #    mask = torch.cat([mask[np.newaxis,...], target[i:i+1],None,...])
    #    post += [mask[np.newaxis,...]]
    #    if i < bs_plot:
    #        mosaic[y:y + h, x:x + w, :] = im
    #Image.fromarray(mosaic).save(fname)

    for i in range(min(bs, 9)):
        im = (images[i].repeat(3,1,1) * 255).to(torch.uint8)
        pred_mask = torch.tensor(plot_polygon(images[i].permute(1,2,0), polygons[i], color=(255,0,0)))
        mask = torch.cat([pred_mask[None,...], target[i:i+1,...]]).to(bool)
        grid += [draw_segmentation_masks(im, mask, alpha=0.5,colors=['red','blue'])[None,]]
        post += [pred_mask[None,]]

    grid = make_grid(torch.cat(grid), nrow=3).permute(1,2,0).numpy()

    if save:
        Image.fromarray(grid).save(fname)

    return torch.cat(post)

    

def plot_semantic_segmentation(inp, pred, target, label_names , fname, save=False):
    label_names = ['background'] + label_names
    class_labels = dict(zip(np.arange(len(label_names)), label_names))
    mask_list = []
    max_show = min(2, inp.shape[0])
    for b in range(max_show):
        im = inp[b].repeat(3,1,1).permute(1,2,0).cpu().numpy()
        if int(im.max()) <= 1:
            im = ( im * 255).astype(int)
        mask_img = wandb.Image(im, masks={
              "predictions": {
                  "mask_data": pred[b].cpu().numpy(),
                          "class_labels": class_labels
                            },
                "ground_truth": {
                    "mask_data": target[b].cpu().numpy(),
                          "class_labels": class_labels
                          },
                            })
        mask_list += [mask_img]


    # save the image 
    grid = []
    im = inp[0].repeat(3,1,1).to(torch.uint8).cpu()
    pred_mask = pred[0].cpu()
    target_mask = target[0].cpu()
    grid += [draw_segmentation_masks(im, pred_mask.bool(), alpha=0.5,colors=['blue','red','green', 'purple'])[None,]]
    grid += [draw_segmentation_masks(im, target_mask.bool(), alpha=0.5,colors=['blue','red','green', 'purple'])[None,]]
    grid = make_grid(torch.cat(grid), nrow=2).permute(1,2,0).numpy()
    if fname[-4:] not in ('.jpg','.png'):
        fname = fname + '.jpg'
    Image.fromarray(grid).save(fname)

    return mask_list
