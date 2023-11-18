from ml_collections import config_dict
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
from torch import nn
from einops import rearrange
from model import Encoder, Decoder
import yaml
import wandb
import numpy as np
from pathlib import Path
from einops import rearrange
#from absl import flags, app
import argparse
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import vocab
from pytorch_lightning.callbacks import LearningRateMonitor
from MEGABYTE_pytorch import MEGABYTE_Encoder, MEGABYTE_Decoder
from generative_cond.discriminator import NLayerDiscriminator, weights_init
from generative_cond.dataloader import LoadPhysioNet
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.utils import make_grid, draw_segmentation_masks
from PIL import Image

#FLAGS = flags.FLAGS
#flags.DEFINE_string('cfg_path', None, 'Config file to use for the task.')


def get_task_and_dataset(config):
    if hasattr(config.task, 'name'):
        task = task_lib.TaskRegistry.lookup(config.task.name)(config.task)
        assert hasattr(config.dataset, 'name') , 'Config dataset should contain name, for single task.'
        dataset = dataset_lib.DatasetRegistry.lookup(config.dataset.name)(config.dataset)

    else:
        task,dataset = [], []
        for k in config.task.keys():
            s = config.task.get(k)
            task += [task_lib.TaskRegistry.lookup(s.name)(s)]
            assert k in config.dataset, 'Dataset key name must match with task.'
            s = config.dataset.get(k)
            dataset += [dataset_lib.DatasetRegistry.lookup(s.name)(s)]

    return task, dataset
    

def wgan_loss(logits_real, logits_fake):
    #return -torch.mean(logits_real) + torch.mean(logits_fake)
    loss_real = logits_real.sum() / (logits_real != 0).sum()
    loss_fake = logits_fake.sum() / (logits_fake != 0).sum()
    return -loss_real + loss_fake
    return -torch.mean(logits_real) + torch.mean(logits_fake)

def hinge_d_loss(logits_real, logits_fake):
    #loss_real = torch.mean(F.leaky_relu(1. - logits_real))
    #loss_fake = torch.mean(F.leaky_relu(1. + logits_fake))
    loss_real = F.leaky_relu(1. - logits_real)
    loss_real = loss_real.sum() / (logits_real != 0).sum()
    loss_fake = F.leaky_relu(1. + logits_fake)
    loss_fake = loss_fake.sum() / (logits_fake != 0).sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def _gradient_penalty(x, descriminator):
    """
    Refer from https://github.com/rosshemsley/gander/blob/main/gander/models/gan.py



    Compute the gradient penalty term used in WGAN-gp.
    Returns the gradient penalty for each batch entry, the loss term is computed as the average.
    Works by sampling points on the line segment between x_r and x_g, then computing the gradient
    of the critic with respect to each sample point.

    Following implementation of WGAN GP, Xr and Xg needed to be interpolated
    https://github.com/Zeleni9/pytorch-wgan/blob/d5b9b4db573f2efbfa56e115d46b28d1f0465312/models/wgan_gradient_penalty.py#L296

        """
    batch_size = x.size(0)
    if x.requires_grad == False:
        x.requires_grad = True

    # We compute the gradient of the parameters using the regular autograd.
    # The key to making this work is including `create_graph`, this means that the computations
    # in this penalty will be added to the computation graph for the loss function, so that the
    # second partial derivatives will be correctly computed.
    f_x = descriminator(x)

    grad = torch.autograd.grad(
                    outputs=f_x,
                    inputs=x,
                    grad_outputs=torch.ones_like(f_x),
                    create_graph=True,
                    only_inputs=True,
                    )[0]
    grad_x_flat = grad.contiguous().view(batch_size, -1)
    gradient_norm = torch.linalg.norm(grad_x_flat, dim=1)
    gp = torch.pow((gradient_norm - 1.0), 2)

    return gp, grad



def _random_sample_line_segment(x1, x2):
    """
    Refer from https://github.com/rosshemsley/gander/blob/main/gander/models/gan.py
    Given two tensors [B,C,H,W] of equal dimensions, in a batch of size B.
    Return a tensor containing B samples randomly sampled on the line segment between each point x1[i], x2[i].
    """
    batch_size = x1.size(0)
    epsilon = _unif(batch_size).type_as(x1)
    return epsilon[:, None, None, None] * x1 + (1 - epsilon)[:, None, None, None] * x2



def _unif(batch_size):
    return torch.distributions.uniform.Uniform(0, 1).sample([batch_size])


class GANModel(pl.LightningModule):

    """
    Trainer to train tasks in language modeling. Our backbone is FocalNet that produce 3 scales, 
    we only take the second scale which is 32, from 512x512 image, similar to ViT patch size 16.
    Also referring to the KaiMing's paper Exploring Plain Vision Transformer Backbones for Object Detection.

    Args:
        encoder: Encoder model
        deocder: Deocder model
        task [List] : A list of tasks that perform preprocessing and postprocessing for specific task

    """
    def __init__(self,learning_rate=1e-3):

        super().__init__()
        #TODO important to log the hyp like bins ( easy to change and never got back the result)
        #self.encoder = Encoder(in_channels = cfg.model.in_channels, 
        #        backbone_weight = None if cfg.model.model_weight is not None else cfg.model.backbone_weight)

        img_size = 256
        p = 64
        self.encoder = MEGABYTE_Encoder(
            num_tokens = 256,
            dim = 256,
            depth = (6, 2),
            max_seq_len = (img_size **2 // p, p), # strictly for image size 256
            flash_attn = False)

        num_tokens = 256+1
        self.decoder = MEGABYTE_Decoder(
            num_tokens = num_tokens, #num_Classes
            dim = 256,
            depth = (2, 2),
            max_seq_len = (img_size ** 2 // p , p), # strictly for image size 256
            flash_attn = False)

        
        self.discriminator = NLayerDiscriminator(input_nc=num_tokens + 1,
                                                 n_layers=5,
                                                 use_actnorm=False,
                                                 ndf=64,
                                                 s =1,
                                                 padw=1,
                                                 kw=3,
                                                 ).apply(weights_init)

        self.healthy_embed = nn.Sequential(nn.Conv2d(1, num_tokens, kernel_size=1), nn.BatchNorm2d(num_tokens)) # 1 is the channel of healthy images

        #######################

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.weight_decay = 0.01
        self.learning_rate = learning_rate
        self.disc_factor = 1.
        self.discriminator_iter_start = 1000
        #self.disc_loss = hinge_d_loss
        self.disc_loss = wgan_loss
        self.disc_act = torch.nn.Sigmoid() if self.disc_loss == hinge_d_loss else torch.nn.Identity()
        self.automatic_optimization = False


    def forward(self, x, seq=None):
        """
        Args:
            x : `tensor` (b t)
        """
        encoded = self.encoder(x)

        if seq is not None: #train
            pred = self.decoder(seq, encoded)
        else:
            pred = self.decoder.generate(encoded, default_batch_size=x.shape[0] * 1024) #encoded[0] * encoder[1]

        
        return pred

    #def training_step(self,batch, batch_idx , optimizer_idx):
    def training_step(self,batch, batch_idx ):
        """
        As long as first argument of out is image, and second argument is boxes.
        """
        g_opt, d_opt = self.optimizers()
        image, masks, healthy = batch
        mask = masks.clone()
        im = image.clone() 
        image = image.cpu() # for plots
        masks = masks.cpu() # for plots
        im *= 255
        b = im.shape[0]
        
        #==============================================
        # if using random_patch
        mask_loc = torch.cat([random_patch(image.shape[-1])[None,] for i in range(mask.shape[0])]).bool() | mask.bool().cpu()
        mask_loc = rearrange(mask_loc, "b  (h p1) (w p2) -> b (h w) (p1 p2)", p1=8, p2=8)
        mask_loc = mask_loc.view(b,-1) 
        #==============================================

        im = rearrange(im, "b c (h p1) (w p2) -> b (c h w) (p1 p2)", p1=8, p2=8)
        mask = rearrange(mask, "b  (h p1) (w p2) -> b (h w) (p1 p2)", p1=8, p2=8)
        im = im.view(b, -1).int()
        mask = mask.view(b,-1) 
        input_seq = mask* 256 #assign to last token, 0-255 is image pixel range

        rand = (torch.rand(mask.shape) * 256).int()
        #mask_loc = (torch.rand(mask.shape) > 0.5) | (mask.bool().cpu())
        #mask_loc = mask.bool().cpu()
        im_aug = torch.where(mask_loc.to(im.device) ,rand.to(im.device) ,im)
        #im_aug = im
        logits = self(im_aug, input_seq)
        #masked loss
        logits = rearrange(logits, 'b l c -> b c l')
        rec_loss = self.criterion(logits, im.long()) * (1.-mask)


        logits = rearrange(logits, "b c (k p) -> b c k p", p=64)
        logits = rearrange(logits, "b c (h w) (p1 p2) -> b c (h p1) (w p2)", h = np.sqrt(logits.shape[2]).astype(int), p1=8, p2=8)

        if self.local_rank == 0:
            if (self.global_step % 500 == 0) & (self.global_step > 1):
                grid = []
                pseudo = torch.argmax(torch.softmax(logits,1),1)[:,None].clamp(0,255).repeat(1,3,1,1).to(torch.uint8).cpu()
                image = (image.repeat(1,3,1,1) * 255).to(torch.uint8)
                masks_draw = masks[:,None].to(bool)
                drawn = [draw_segmentation_masks(image[i], masks_draw[i], alpha=0.5,colors=['red','blue']) for i in range(2)]
                grid = [drawn[0], pseudo[0], drawn[1], pseudo[1]]
                grid = make_grid(grid, nrow=2).permute(1,2,0).numpy()

                self.logger.log_image( key=f"train_batch", images=[ wandb.Image(Image.fromarray(grid))] )

            #    self.log_dict({'Dice':result})

        #self.log("my_lr", self.trainer.lr_scheduler_configs.scheduler.get_last_lr()[0], prog_bar=True, on_step=True)

        #================================================================
        #     Generator
        #================================================================
        #if optimizer_idx == 0:
        #logits_fake = self.discriminator(logits.detach())
        #logits_fake = self.disc_act(self.discriminator(torch.cat([logits , masks.to(logits.dtype).to(logits.device)[:,None]],1)))
        logits_fake = self.disc_act(self.discriminator(torch.cat([logits , masks.to(logits.dtype).to(logits.device)[:,None]],1))) * masks.to(logits.device).to(logits.device)[:,None]
        disc_factor = adopt_weight(self.disc_factor, self.global_step, threshold=self.discriminator_iter_start)
        #g_loss = disc_factor * -torch.mean(logits_fake) + torch.mean(rec_loss)
        g_loss = disc_factor * -(logits_fake.sum()/ (logits_fake != 0).sum()) + torch.mean(rec_loss) * 100
        self.log_dict({'Reconstruction': g_loss}, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()
        #return g_loss


        #================================================================
        #     Discriminator
        #================================================================
        #if optimizer_idx == 1:
        # discriminator
        healthy_emb = self.healthy_embed(healthy)
        logits_real = self.disc_act(self.discriminator(torch.cat([healthy_emb,masks.to(logits.dtype).to(logits.device)[:,None]],1))) * masks.to(logits.dtype).to(logits.device)[:,None]
        logits_fake = self.disc_act(self.discriminator(torch.cat([logits.detach(), masks.to(logits.dtype).to(logits.device)[:,None]],1))) * masks.to(logits.dtype).to(logits.device)[:,None]

        #logits_real = self.discriminator(healthy_emb)
        #logits_fake = self.discriminator(logits.detach())

        disc_factor = adopt_weight(self.disc_factor, self.global_step, threshold=self.discriminator_iter_start)
        if self.disc_loss == wgan_loss:
            x_hat = _random_sample_line_segment(
                            healthy_emb.detach(),
                            logits.detach()
                    )
            gp, _ = _gradient_penalty(torch.cat([x_hat, masks.to(logits.dtype).to(logits.device)[:,None]],1), nn.Sequential(self.discriminator, self.disc_act))
            #gp, _ = _gradient_penalty(torch.cat([x_hat, masks.to(logits.dtype).to(logits.device)[:,None]],1), self.discriminator)
            #gp, _ = _gradient_penalty(x_hat, self.discriminator)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake) +  gp.mean()
        else:
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        if batch_idx % 3 == 0:
            d_opt.zero_grad()
            self.manual_backward(d_loss)
            d_opt.step()
            self.log_dict({'Discriminator': d_loss, 'Logits Fake':logits_fake.mean(),'Logits Real':logits_real.mean() }, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        #return d_loss



    def on_train_epoch_end(self):
        pass


    def validation_step(self,batch, batch_idx):
        #https://github.com/Lightning-AI/lightning/discussions/13041
        out  = self.dataset.process(batch) # im ,target,...
        target = out[1]
        im = self.task.preprocess_target(out)['image']
        pred_seq , _ = self.predict(im)
        if im.max().int() <= 1:
            im = ( im * 255).int()
        post = self.task.postprocess( im , target , pred_seq , self.dataset, fname='plots/images_val.jpg')
        if isinstance(post, dict):
            result = post['result']
            plots = post['plot']
            if (batch_idx == 0 ) & (self.local_rank == 0):
                self.logger.experiment.log( {f"val_batch": plots} )
        else:
            if (batch_idx == 0 ) & (self.local_rank == 0):
                self.logger.log_image( key=f"val_batch", images=[ wandb.Image("plots/images_val.jpg")] )



        return result

    def test_step(self,batch, batch_idx):
        #https://github.com/Lightning-AI/lightning/discussions/13041
        out  = self.dataset.process(batch) # im ,target,...
        target = out[1]
        im = self.task.preprocess_target(out)['image']

        pred_seq , _ = self.predict(im)
        if im.max().int() <= 1:
            im = ( im * 255).int()
        post = self.task.postprocess( im , target , pred_seq , self.dataset)
        if isinstance(post, dict):
            result = post['result']
            plots = post['plot']
            if (batch_idx == 0 ) & (self.local_rank == 0):
                self.logger.experiment.log( {f"test_batch": plots} )
        else:
            if batch_idx == 0:
                self.logger.log_image( key=f"test_batch", images=[ wandb.Image("images_train.jpg")] )

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

    
    def validation_epoch_end(self, output):
        results = self.task.compute()
        self.log_dict({'Val_Dice':results})
        self.task.reset()

    def test_epoch_end(self, output):
        results = self.task.compute()
        self.log_dict({'Test_Dice':results})
        self.task.reset()

    def configure_optimizers(self):

        param_dicts = [

                {
                    "params" : self.encoder.parameters()
                },
                {
                    "params" : self.decoder.parameters()
                },

                ]
        opt_gen = torch.optim.AdamW(param_dicts, lr=self.learning_rate,
                                              weight_decay= self.weight_decay)

        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=self.learning_rate, betas=(0.5, 0.9))

        return [opt_gen, opt_disc], []




    def predict(self, images):
        """
        To perform inference on image. Infer return prediction in normalised xyxy

        Args:
          images: `float` tensor of (bsz, c, h, w).
          prompt_seq: `int` sequence visible to the model of shape (bsz, seqlen),
            or (bsz, instances, seqlen) if there are multiple sequences per image.
        """
        out = self.dataset.process((images, None)) # different task may return different number of outputs, (im, target,...)
        bsz = images.shape[0]
        x = images.reshape(bsz, -1)
        seq = (out[0] * 255).int()
        seq = seq.reshape(bsz, -1)
        pred_seq = self(seq)
        #prompt_seq = build_prompt_seq_from_task_id( self.task.task_vocab_id, prompt_shape=(bsz,1 ))
        #encoded = self.encoder(images)
        #pred_seq, logits = self.decoder.infer(prompt = prompt_seq, 
        #                   memory = encoded,
        #                   max_seq_len = (self.cfg.task.max_instances_per_image_test * 5 ),
        #                   temperature = self.cfg.task.temperature,
        #                   top_k=self.cfg.task.top_k, top_p=self.cfg.task.top_p
        #                   )
        return pred_seq, None

        

class NonGanModel(pl.LightningModule):

    """
    Trainer to train tasks in language modeling. Our backbone is FocalNet that produce 3 scales, 
    we only take the second scale which is 32, from 512x512 image, similar to ViT patch size 16.
    Also referring to the KaiMing's paper Exploring Plain Vision Transformer Backbones for Object Detection.

    Args:
        encoder: Encoder model
        deocder: Deocder model
        task [List] : A list of tasks that perform preprocessing and postprocessing for specific task

    """
    def __init__(self,learning_rate=1e-3):

        super().__init__()
        #TODO important to log the hyp like bins ( easy to change and never got back the result)
        #self.encoder = Encoder(in_channels = cfg.model.in_channels, 
        #        backbone_weight = None if cfg.model.model_weight is not None else cfg.model.backbone_weight)

        img_size = 256
        p = 64
        self.encoder = MEGABYTE_Encoder(
            num_tokens = 256,
            dim = 256,
            depth = (6, 2),
            max_seq_len = (img_size **2 // p, p), # strictly for image size 256
            flash_attn = False)

        num_tokens = 256+1
        self.decoder = MEGABYTE_Decoder(
            num_tokens = num_tokens, #num_Classes
            dim = 256,
            depth = (2, 2),
            max_seq_len = (img_size ** 2 // p , p), # strictly for image size 256
            flash_attn = False)

        
        self.discriminator = NLayerDiscriminator(input_nc=num_tokens , #if GANModel +1
                                                 n_layers=5,
                                                 use_actnorm=False,
                                                 ndf=64,
                                                 s =1,
                                                 padw=1,
                                                 kw=3,
                                                 ).apply(weights_init)

        self.healthy_embed = nn.Sequential(nn.Conv2d(1, num_tokens, kernel_size=1), nn.BatchNorm2d(num_tokens)) # 1 is the channel of healthy images

        #######################

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.weight_decay = 0.01
        self.learning_rate = learning_rate
        self.disc_factor = 1.
        self.discriminator_iter_start = 10 #1000
        #self.disc_loss = hinge_d_loss
        self.disc_loss = wgan_loss
        self.disc_act = torch.nn.Sigmoid() if self.disc_loss == hinge_d_loss else torch.nn.Identity()
        self.automatic_optimization = False


    def forward(self, x, seq=None):
        """
        Args:
            x : `tensor` (b t)
        """
        encoded = self.encoder(x)

        if seq is not None: #train
            pred = self.decoder(seq, encoded)
        else:
            pred = self.decoder.generate(encoded, default_batch_size=x.shape[0] * 1024) #encoded[0] * encoder[1]

        
        return pred

    #def training_step(self,batch, batch_idx , optimizer_idx):
    def training_step(self,batch, batch_idx ):
        """
        As long as first argument of out is image, and second argument is boxes.
        """
        g_opt, d_opt = self.optimizers()
        image, masks, healthy = batch
        #mask = masks.clone()
        #im = image.clone() 
        im = torch.cat([image, healthy])
        del healthy
        mask = torch.cat([masks, torch.zeros(image.shape[0],masks.shape[1], masks.shape[2], dtype=masks.dtype).to(masks.device)])

        image = image.cpu() # for plots
        masks = masks.cpu() # for plots
        im *= 255
        b = im.shape[0]
        
        mask_loc = torch.cat([random_patch(image.shape[-1])[None,] for i in range(mask.shape[0])]).bool() | mask.bool().cpu()

        im = rearrange(im, "b c (h p1) (w p2) -> b (c h w) (p1 p2)", p1=8, p2=8)
        mask = rearrange(mask, "b  (h p1) (w p2) -> b (h w) (p1 p2)", p1=8, p2=8)

        #==============================================
        # if using random_patch
        mask_loc = rearrange(mask_loc, "b  (h p1) (w p2) -> b (h w) (p1 p2)", p1=8, p2=8)
        mask_loc = mask_loc.view(b,-1) 
        #==============================================

        im = im.view(b, -1).int()
        mask = mask.view(b,-1) 
        input_seq = mask* 256 #assign to last token, 0-255 is image pixel range

        #==============================
        
        #==============================
        rand = (torch.rand(mask.shape) * 256).int()
        mask_loc = (torch.rand(mask_loc.shape) > 0.5) | (mask_loc.bool().cpu())
        #mask_loc = mask.bool().cpu()
        im_aug = torch.where(mask_loc.to(im.device) ,rand.to(im.device) ,im)
        #im_aug = im
        logits = self(im_aug, input_seq)
        
        #masked loss
        logits = rearrange(logits, 'b l c -> b c l')
        rec_loss = self.criterion(logits, im.long()) * (1.-mask)


        logits = rearrange(logits, "b c (k p) -> b c k p", p=64)
        logits = rearrange(logits, "b c (h w) (p1 p2) -> b c (h p1) (w p2)", h = np.sqrt(logits.shape[2]).astype(int), p1=8, p2=8)

        if self.local_rank == 0:
            if (self.global_step % 2000 == 0) & (self.global_step > 1):
                grid = []
                pseudo = torch.argmax(torch.softmax(logits,1),1)[:,None].clamp(0,255).repeat(1,3,1,1).to(torch.uint8).cpu()
                image = (image.repeat(1,3,1,1) * 255).to(torch.uint8)
                masks_draw = masks[:,None].to(bool)
                drawn = [draw_segmentation_masks(image[i], masks_draw[i], alpha=0.5,colors=['red','blue']) for i in range(2)]
                grid = [drawn[0], pseudo[0], drawn[1], pseudo[1]]
                grid = make_grid(grid, nrow=2).permute(1,2,0).numpy()

                self.logger.log_image( key=f"train_batch", images=[ wandb.Image(Image.fromarray(grid))] )

            #    self.log_dict({'Dice':result})

        #self.log("my_lr", self.trainer.lr_scheduler_configs.scheduler.get_last_lr()[0], prog_bar=True, on_step=True)

        #================================================================
        #     Generator
        #================================================================
        ##if optimizer_idx == 0:
        ##logits_fake = self.disc_act(self.discriminator(torch.cat([logits , masks.to(logits.dtype).to(logits.device)[:,None]],1)))
        ##logits_fake = self.disc_act(self.discriminator(torch.cat([logits , masks.to(logits.dtype).to(logits.device)[:,None]],1))) * masks.to(logits.device).to(logits.device)[:,None]
        #logits_fake = self.disc_act(logits)
        #disc_factor = adopt_weight(self.disc_factor, self.global_step, threshold=self.discriminator_iter_start)
        ##g_loss = disc_factor * -torch.mean(logits_fake) + torch.mean(rec_loss)
        #g_loss = disc_factor * -(logits_fake.sum()/ (logits_fake != 0).sum()) + torch.mean(rec_loss) * 100
        g_loss = torch.mean(rec_loss)
        self.log_dict({'Reconstruction': g_loss}, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()
        #return g_loss


        ##================================================================
        ##     Discriminator
        ##================================================================
        ##if optimizer_idx == 1:
        ## discriminator
        #healthy_emb = self.healthy_embed(healthy)
        #logits_real = self.disc_act(healthy_emb)
        #logits_fake = self.disc_act(logits.detach())
        ##logits_real = self.disc_act(self.discriminator(torch.cat([healthy_emb,masks.to(logits.dtype).to(logits.device)[:,None]],1))) * masks.to(logits.dtype).to(logits.device)[:,None]
        ##logits_fake = self.disc_act(self.discriminator(torch.cat([logits.detach(), masks.to(logits.dtype).to(logits.device)[:,None]],1))) * masks.to(logits.dtype).to(logits.device)[:,None]

        #disc_factor = adopt_weight(self.disc_factor, self.global_step, threshold=self.discriminator_iter_start)
        #if self.disc_loss == wgan_loss:
        #    x_hat = _random_sample_line_segment(
        #                    healthy_emb.detach(),
        #                    logits.detach()
        #            )
        #    #gp, _ = _gradient_penalty(torch.cat([x_hat, masks.to(logits.dtype).to(logits.device)[:,None]],1), nn.Sequential(self.discriminator, self.disc_act))
        #    #gp, _ = _gradient_penalty(torch.cat([x_hat, masks.to(logits.dtype).to(logits.device)[:,None]],1), self.discriminator)
        #    gp, _ = _gradient_penalty(x_hat, nn.Sequential(self.discriminator, self.disc_act))
        #    d_loss = disc_factor * self.disc_loss(logits_real, logits_fake) +  gp.mean()
        #else:
        #    d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        #if batch_idx % 3 == 0:
        #    d_opt.zero_grad()
        #    self.manual_backward(d_loss)
        #    d_opt.step()
        #    self.log_dict({'Discriminator': d_loss, 'Logits Fake':logits_fake.mean(),'Logits Real':logits_real.mean() }, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        ##return d_loss



    def on_train_epoch_end(self):
        pass


    def validation_step(self,batch, batch_idx):
        #https://github.com/Lightning-AI/lightning/discussions/13041
        out  = self.dataset.process(batch) # im ,target,...
        target = out[1]
        im = self.task.preprocess_target(out)['image']
        pred_seq , _ = self.predict(im)
        if im.max().int() <= 1:
            im = ( im * 255).int()
        post = self.task.postprocess( im , target , pred_seq , self.dataset, fname='plots/images_val.jpg')
        if isinstance(post, dict):
            result = post['result']
            plots = post['plot']
            if (batch_idx == 0 ) & (self.local_rank == 0):
                self.logger.experiment.log( {f"val_batch": plots} )
        else:
            if (batch_idx == 0 ) & (self.local_rank == 0):
                self.logger.log_image( key=f"val_batch", images=[ wandb.Image("plots/images_val.jpg")] )



        return result

    def test_step(self,batch, batch_idx):
        #https://github.com/Lightning-AI/lightning/discussions/13041
        out  = self.dataset.process(batch) # im ,target,...
        target = out[1]
        im = self.task.preprocess_target(out)['image']

        pred_seq , _ = self.predict(im)
        if im.max().int() <= 1:
            im = ( im * 255).int()
        post = self.task.postprocess( im , target , pred_seq , self.dataset)
        if isinstance(post, dict):
            result = post['result']
            plots = post['plot']
            if (batch_idx == 0 ) & (self.local_rank == 0):
                self.logger.experiment.log( {f"test_batch": plots} )
        else:
            if batch_idx == 0:
                self.logger.log_image( key=f"test_batch", images=[ wandb.Image("images_train.jpg")] )

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

    
    def validation_epoch_end(self, output):
        results = self.task.compute()
        self.log_dict({'Val_Dice':results})
        self.task.reset()

    def test_epoch_end(self, output):
        results = self.task.compute()
        self.log_dict({'Test_Dice':results})
        self.task.reset()

    def configure_optimizers(self):

        param_dicts = [

                {
                    "params" : self.encoder.parameters()
                },
                {
                    "params" : self.decoder.parameters()
                },

                ]
        opt_gen = torch.optim.AdamW(param_dicts, lr=self.learning_rate,
                                              weight_decay= self.weight_decay)

        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=self.learning_rate, betas=(0.5, 0.9))

        return [opt_gen, opt_disc], []




    def predict(self, images):
        """
        To perform inference on image. Infer return prediction in normalised xyxy

        Args:
          images: `float` tensor of (bsz, c, h, w).
          prompt_seq: `int` sequence visible to the model of shape (bsz, seqlen),
            or (bsz, instances, seqlen) if there are multiple sequences per image.
        """
        out = self.dataset.process((images, None)) # different task may return different number of outputs, (im, target,...)
        bsz = images.shape[0]
        x = images.reshape(bsz, -1)
        seq = (out[0] * 255).int()
        seq = seq.reshape(bsz, -1)
        pred_seq = self(seq)
        #prompt_seq = build_prompt_seq_from_task_id( self.task.task_vocab_id, prompt_shape=(bsz,1 ))
        #encoded = self.encoder(images)
        #pred_seq, logits = self.decoder.infer(prompt = prompt_seq, 
        #                   memory = encoded,
        #                   max_seq_len = (self.cfg.task.max_instances_per_image_test * 5 ),
        #                   temperature = self.cfg.task.temperature,
        #                   top_k=self.cfg.task.top_k, top_p=self.cfg.task.top_p
        #                   )
        return pred_seq, None


def random_patch( im_size):
    center = im_size // 2
    n_box = 8
    mask = torch.zeros(im_size, im_size)
    loc = [ torch.clamp(torch.randn(2),-1,1) for _ in range(n_box)]
    ratio = [torch.rand(2) for _ in range(n_box)]
    for (x_delt,y_delt),(w,h) in zip(loc,ratio):
        x,y = int(center + x_delt * center), int(center+ y_delt * center)
        w = int(w * (center//3))
        h = int(h * (center//3))
        mask[(y-w): (y+w),(x-w):(x+w)] = 1
    return mask




def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='Path to config file')
    parser.add_argument('--device', type=str, default='0', help='GPU device to use.')
    parser.add_argument('--wandb', action='store_true', help='Online logging.')
    parser.add_argument('--mode', type=str, help='train or test.')
    parser.add_argument('--ckpt',type=str, help='Trained model')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':


    opt = parse_opt()


    logger_opt = {}
    logger_opt['project'] = "Pix2Seq"
    logger_opt['entity'] = 'chunhunghow'
    logger_opt['dir'] = Path('./wandb') / "Pix2Seq"
    logger_opt['name'] = 'Conditional generation'

    if len(opt.device) > 1: #DP training only online 
        wandb.init(settings=wandb.Settings(start_method='fork'), **logger_opt)
        logger = WandbLogger()
    else:
        logger_opt['offline'] = not opt.wandb
        logger = WandbLogger(**logger_opt)


    #model = Model(cfg , None, None)
    #task,dataset = get_task_and_dataset(cfg)

    data_mode = 'abnormal'
    data_path = lambda x: f'/home/data_repo/INSTANCE_ICH/data2D/images/{x}'
    #unhealthy_dataset = LoadPhysioNet(data_path('train'), mode=mode, label=mode == 'abnormal', img_size=256) #only abnormal has mask
    batch_size = 3
    model = NonGanModel()
    #model = GANModel()
    loader = DataLoader(
        LoadPhysioNet(data_path(opt.mode), mode=data_mode, label= data_mode == 'abnormal', img_size=256),
        batch_size=batch_size,
        num_workers=4,
        drop_last=False)

    if opt.ckpt is not None:
        model.load_state_dict(torch.load(opt.ckpt , map_location= 'cpu')['state_dict'])
        print(f'Loaded weights from {opt.ckpt}')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    device = [int(opt.device)] if (len(opt.device) == 1 ) else [int(k) for k in opt.device.split(',')]
    trainer = Trainer(accelerator="gpu", devices=device, logger=logger, max_epochs=500)
    if opt.mode == 'train':
        #trainer.fit(model, dataset.load_dataloader('train'), dataset.load_dataloader('val'))
        trainer.fit(model, loader)
        #trainer.fit(model, dataset.load_dataloader('train'))
    elif opt.mode == 'test':
        trainer.test(model, loader)
    else:
        raise ValueError('Should specify train or test as mode.')
    
    
    
