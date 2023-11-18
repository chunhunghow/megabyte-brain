from ml_collections import config_dict
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
from torch import nn
from einops import rearrange
from util import create_mask, build_prompt_seq_from_task_id, decode_object_seq_to_bbox, LossFunction, seq_to_bbox
#from model import Encoder, Decoder
from tasks import task as task_lib
from data import dataset as dataset_lib
from data import datasets
from tasks import object_detection
from tasks import classification
from tasks import semantic_segmentation
from tasks import instance_segmentation 
from data.multitask import MultiDataset
from tasks.multitask import MultiTask
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
import glob
from MEGABYTE_pytorch import MEGABYTE_Encoder, MEGABYTE_Decoder


#FLAGS = flags.FLAGS
#flags.DEFINE_string('cfg_path', None, 'Config file to use for the task.')


def get_task_and_dataset(config):
    if hasattr(config.task, 'name'):
        task = task_lib.TaskRegistry.lookup(config.task.name)(config.task)
        task.cfg.update(config.model)
        assert hasattr(config.dataset, 'name') , 'Config dataset should contain name, for single task.'
        dataset = dataset_lib.DatasetRegistry.lookup(config.dataset.name)(config.dataset)

    else:
        task,dataset = [], []
        for k in config.task.keys():
            s = config.task.get(k)
            s.update(config.model)
            task += [task_lib.TaskRegistry.lookup(s.name)(s)]
            assert k in config.dataset, 'Dataset key name must match with task.'
            s = config.dataset.get(k)
            dataset += [dataset_lib.DatasetRegistry.lookup(s.name)(s)]

        dataset = MultiDataset(dataset)
        task = MultiTask(task)
    
    return task, dataset
    



class Model(pl.LightningModule):

    """
    Also referring to the KaiMing's paper Exploring Plain Vision Transformer Backbones for Object Detection.
    The decoder input has taskID as first token, for semantic segmentation, we can give taskID and all zeros, 
    for lesion segmentation, we provide taskID and bounding boxes as input, for classification we provide taskID and all zeros,



    Args:
        encoder: MegaByte Encoder model
        deocder: MegaByte Deocder model
        task [List] or Object : A list of tasks or single task object 
                                that perform preprocessing and postprocessing for specific task

    """
    def __init__(self,cfg, task, dataset, learning_rate=1e-3):
        super().__init__()

        img_size = cfg.model.size
        p = cfg.model.patch_size_sqrt ** 2

        # encoder only takes in im with 0-255, num_tokens = 256
        self.encoder = MEGABYTE_Encoder(
            num_tokens = 256,
            dim = cfg.model.hidden_dim,
            depth = (6, 2),
            max_seq_len = (img_size **2 // p, p), # strictly for image size 256 and 1 channel, due to computation memory
            flash_attn = False)

        # for multitask, suggest to have 2000 tokens, 0-100 for taskID and special token, 100-1000 for classification, 1000 - 2000 for bounding box location
        # classification can be image level, can also be pixel level segmentation, like lesion segmentation 

        self.decoder = MEGABYTE_Decoder(
            #num_tokens = 1+len(cfg.dataset['train']['label']), #num_Classes
            num_tokens= cfg.model.vocab_size,
            dim = cfg.model.hidden_dim, # limited by computation memory
            depth = (2, 2),
            max_seq_len = (img_size ** 2 // p , p), # strictly for image size 256
            # max_seq_len is set to image length / patch size, should be more than enough for bounding boxes with contrastive noise < 512 length
            flash_attn = False)

        
        if cfg.model.model_weight is not None:
            self.load_state_dict(torch.load(cfg.model.model_weight, map_location='cpu')['state_dict'])
            print(f'### Loaded model weight from {cfg.model.model_weight}  ###')
        self.task = task
        self.dataset = dataset
        #self.criterion = LossFunction(cfg.task.loss) 
        self.criterion = LossFunction(cfg.model.loss) # fix loss for all tasks
        self.weight_decay = 0.01
        self.cfg = cfg
        self.save_hyperparameters()
        self.task.reset()
        self.maxrank = 0


    def forward(self, x, seq):
        """
        Args:
        """
        encoded = self.encoder(x)
        if seq.shape[-1] != 1:
            pred = self.decoder(seq, encoded)
        else:
            pred = self.decoder.generate(memory= encoded, prime = seq, default_batch_size=1024)
            
        
        return pred

    def training_step(self,batch, batch_idx):
        """
  
        """
        out = self.dataset.process(batch) # different task may return different number of outputs, (im, target,...)
        batched_examples = self.task.preprocess_target(out)
        im, input_seq, target_seq, token_weights = \
         self.task.preprocess_batched(batched_examples,training=True)
        #im = im.view(im.shape[0],-1)
        if im.max().int() <= 1:
            im = (im * 255).int()

        logits = self(im, input_seq)
        logits = rearrange(logits, 'b l c -> b c l')
        loss = self.criterion(logits, target_seq)
        loss = (loss * token_weights).sum() / token_weights.sum() #padded and eos are less important
        
        
        #if self.global_rank == 0:
        #    self.log_dict(
        #            {'Training Loss': loss}
        #        )
        #    if self.maxrank == 0:
        #        if (self.global_step % 5000 == 0) & (self.global_step > 1):
        #            post = self.task.postprocess( im, [t[1] for t in out] if isinstance(self.task, MultiTask) else out[1] ,logits, self.dataset)
        #            if isinstance(self.task,MultiTask):
        #                self.log_images(post)
        #            else:
        #                self.log_images((0, post))


        self.log("my_lr", self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0], prog_bar=True, on_step=True)
        return loss


    def on_train_epoch_end(self):
        #results = self.task.compute()
        #self.log_tasks(results, 'train')
        self.task.reset()

    def validation_step(self,batch, batch_idx, dataloader_idx):
        #https://github.com/Lightning-AI/lightning/discussions/13041

        out = self.dataset.process(batch, dataloader_idx)
        batched_examples = self.task.preprocess_target(out, dataloader_idx)
        im, input_seq, target_seq, token_weights = \
         self.task.preprocess_batched(batched_examples,training=True, idx=dataloader_idx)
        logits  = self.predict(im, input_seq)
        if batch_idx == 0:
            post = self.task.postprocess( im, out[1] ,logits, self.dataset, fname='val', idx=dataloader_idx) #out[1] is target
            self.log_images((dataloader_idx,post),'val')

        else:
            post = self.task.postprocess( im, out[1] ,logits, self.dataset, save=False, idx=dataloader_idx)

        return post

    def validation_epoch_end(self, out):
        results = self.task.compute()
        self.log_tasks(results, 'val')
        self.task.reset()

    def test_step(self,batch, batch_idx, dataloader_idx=None):

        out = self.dataset.process(batch, idx=dataloader_idx)
        batched_examples = self.task.preprocess_target(out, idx=dataloader_idx)
        im, input_seq, target_seq, token_weights = \
         self.task.preprocess_batched(batched_examples,training=True, idx=dataloader_idx)
        logits  = self.predict(im, input_seq)
        if batch_idx % 10 == 0:
            post = self.task.postprocess( im, out[1] ,logits, self.dataset, fname='test', idx=dataloader_idx) #out[1] is target
            self.log_images((dataloader_idx,post),'test')

        else:
            post = self.task.postprocess( im, out[1] ,logits, self.dataset, save=False, idx=dataloader_idx)
        return post


    def test_epoch_end(self, out):
        results = self.task.compute()
        self.log_tasks(results, mode='test')
        self.task.reset()



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

    

    def configure_optimizers(self):
        param_dicts = [

                {
                    "params" : self.encoder.parameters()
                },
                {
                    "params" : self.decoder.parameters()
                },

                ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.hparams.learning_rate,
                                              weight_decay=self.weight_decay)

        dataloader_len = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        warm_up_epoch = int(self.trainer.max_epochs * 0.1)
        lambda0 = lambda cur_epoch : cur_epoch / warm_up_epoch + 1e-8 if cur_epoch < warm_up_epoch else (1 - (cur_epoch - warm_up_epoch )/ (self.trainer.max_epochs - warm_up_epoch))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        return [optimizer], [lr_scheduler]


    def predict(self, im, input_seq):
        """
        Input will only be single task, with task ID.

        Args:
        """
        if im.max().int() <= 1:
            im = (im * 255).int()

        # not many tasks are sampling task (image generation, conditional generation)
        # for deterministic task like classification, instance smemantic seg, we do not need to draw one by one
         
        if any(input_seq[:,0] == 1):
            input_seq = torch.tensor([[1]]).repeat(im.shape[0],1).to(im.device)
        logits = self(im, input_seq)
        logits = rearrange(logits, 'b l c -> b c l')

        return logits



    def log_images(self , post, mode='train'):

        if isinstance(post, list):
            task_name = [t.cfg.name for t in self.task.tasks]
            for i,p in enumerate(post):
                if isinstance(p, dict):
                    result = p['result']
                    plots = p['plot']
                    self.logger.experiment.log( {f"{task_name[i]}_{mode}": plots} )
                else:
                    for n in glob.glob('output/*.jpg'):
                        t = task_name[i]
                        if (t in n) & (mode in n):
                            self.logger.log_image( key=t + f'_{mode}', images=[ wandb.Image(n)] )

        else: #single task or val, test\
            # post must be a tuple, (1, post) where the first index is dataloader index
            i, p = post
            t = self.task.tasks[i].cfg.name if hasattr(self.task, 'tasks') else self.task.cfg.name
            if isinstance(p, dict):
                result = p['result']
                plots = p['plot']
                self.logger.experiment.log( {f"{t}_{mode}": plots} )
            else:
                for n in glob.glob('output/*.jpg'):
                    if (t in n) & (mode in n):
                        self.logger.log_image( key=t + f'_{mode}', images=[ wandb.Image(n)] )
                #self.logger.log_image( key=f"{mode}_batch", images=[ wandb.Image(f"output/images_{mode}.jpg")] )


    def log_tasks(self, result, mode='train'):
        result = self.task.log_result(result,  dataset_obj=self.dataset, mode=mode)
        if isinstance(self.task,MultiTask):
            for t in result:
                if isinstance(result[t], wandb.Table):
                    self.logger.log_metrics({mode+'_'+t : result[t]})
                elif isinstance(result[t], dict):
                    self.log_dict(result[t])
                else:
                    raise NotImplementedError
        else:
            if isinstance(result, wandb.Table):
                self.logger.log_metrics({mode+'_'+self.task.cfg.name + '_' + self.dataset.cfg.name : result})
            else:
                self.log_dict(result)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='Path to config file')
    parser.add_argument('--device', type=str, default='0', help='GPU device to use.')
    parser.add_argument('--wandb', action='store_true', help='Online logging.')
    parser.add_argument('--mode', type=str, help='train or test.')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':


    opt = parse_opt()
    with open(opt.cfg, 'r') as f:
        cfg = config_dict.ConfigDict(yaml.safe_load(f))

    logger_opt = {}
    logger_opt['project'] = "MegaByte-Pix2Seq"
    logger_opt['entity'] = 'chunhunghow'
    logger_opt['dir'] = Path('./wandb') / "MegaByte-Pix2Seq"
    if opt.mode == 'test':
        if hasattr(cfg.task, 'name'):
            logger_opt['name'] = f"(Test) {cfg.task.name}_{cfg.dataset.name}"
        else:
            mt_name = (list(zip(  [cfg.task[t].name for t in cfg.task.keys()] ,
                     [cfg.dataset[d].name for d in cfg.dataset.keys()])
                     ))
            mt_name = '_'.join([n[0]+'-'+n[1] for n in mt_name])
            logger_opt['name'] = '(Test) ' + mt_name
    else:
        if hasattr(cfg.task, 'name'):
            logger_opt['name'] = f"{cfg.task.name}_{cfg.dataset.name}"
        else:
            mt_name = (list(zip(  [cfg.task[t].name for t in cfg.task.keys()] ,
                     [cfg.dataset[d].name for d in cfg.dataset.keys()])
                     ))
            mt_name = '_'.join([n[0]+'-'+n[1] for n in mt_name])
            logger_opt['name'] = mt_name


    if len(opt.device) > 1: #DP training only online 
        wandb.init(settings=wandb.Settings(start_method='fork'), **logger_opt)
        logger = WandbLogger()
    else:
        logger_opt['offline'] = not opt.wandb
        logger = WandbLogger(**logger_opt)


    task,dataset = get_task_and_dataset(cfg)
    model = Model(cfg , task, dataset)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    device = [int(opt.device)] if (len(opt.device) == 1 ) else [int(k) for k in opt.device.split(',')]
    trainer = Trainer(accelerator="gpu", devices=device, logger=logger, max_epochs=cfg.train.epoch, callbacks=[lr_monitor])
    if opt.mode == 'train':
        #trainer.fit(model, dataset.load_dataloader('train'), val_dataloaders=dataset.load_dataloader('val'))
        trainer.fit(model, dataset.load_dataloader('train'))
        trainer.test(model, dataset.load_dataloader('test'))
    elif opt.mode == 'test':
        trainer.test(model, dataset.load_dataloader('test'))
    else:
        raise ValueError('Should specify train or test as mode.')
    
    
    
