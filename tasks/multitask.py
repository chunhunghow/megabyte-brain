


from typing import Any, Dict, List
import torch
from tasks import task as task_lib
import util
import vocab
import random
import numpy as np
from data.data_utils import augment_bbox


#@task_lib.TaskRegistry.register('multitask')
class MultiTask():

    def __init__(self, tasks):
        """
        Args:
            tasks: `List` task object
        """
        self.tasks = tasks
     
    def reset(self):
        for t in self.tasks:
            t.reset()


    def preprocess_target(self, batch_targets, idx=-1):
        """
        batch_targets : List[List[Dict]]
        idx: -1 means all tasks, otherwise specified

        Return:
        `List` preprocessed targets for corresponding tasks.
        """
        if idx >=0 :
            return self.tasks[idx].preprocess_target(batch_targets)
        assert len(batch_targets) == len(self.tasks) , f"Sample list {len(batch_targets)} should be the same as tasks list (length {len(self.tasks)}). [task1, task2] , [sample1 , sample2] "

        task_process_data = []
        for t in range(len(self.tasks)):
            task_process_data += [self.tasks[t].preprocess_target(batch_targets[t])]

        return task_process_data
        
                
    def preprocess_batched(self, batched_examples , training, idx=-1):
        """
       Typical operations in this preprocessing step for detection task:
       - Quantization and serialization of object instances.
       - Creating the input sequence, target sequence, and token weights.

       Args:
           batched_examples: tuples of feature and label tensors that are
                     preprocessed, batched, and stored with `dict`. FasteRCNN, DINO style.
           training: bool.
           idx: -1 means all tasks, otherwise specified

       Returns:

           images: `float` of shape (bsz, h, w, c)
           input_seq: `int` of shape (bsz, seqlen).
           target_seq: `int` of shape (bsz, seqlen).
           token_weights: `float` of shape (bsz, seqlen)

           Each of the above output become a list ( of number of task / dataset). 

        """

        if idx >=0:
            return self.tasks[idx].preprocess_batched(batched_examples, training)

        task_process_data = []
        for t in range(len(self.tasks)):
            task_process_data += [self.tasks[t].preprocess_batched(batched_examples[t] ,training)]

        # to keep track of the data sample of each task (for postprocess purpose)
        self._track = np.cumsum([d[1].shape[0] for d in task_process_data])
        task_process_data =  [torch.cat(ls) for ls in zip(*task_process_data)]
        return task_process_data



    def postprocess(self, im , target, logits ,dataset_obj, fname='train', save=True, idx=-1):


        fname = ['output/' +self.tasks[t].cfg.name + f'_multitask_{fname}.jpg' for t in range(len(self.tasks))]

        if idx >= 0:
            return self.tasks[idx].postprocess(im, target, logits, dataset_obj.datasets[idx], fname[idx], save=save)

        last = 0
        post = []
        for i, sz in enumerate(self._track):
            if im[last:sz].shape[0] == 0:
                continue
            post += [self.tasks[i].postprocess(im[last: sz], target[i], logits[last:sz], dataset_obj.datasets[i], fname[i], save=save)]
            last = sz

        return post



    def compute(self):
        results = []
        for t in range(len(self.tasks)):
            results += [self.tasks[t].compute()]
        return results


    def log_result(self, results,dataset_obj, mode='val'):
        """
        wandb_obj can be Table or dictionary.
        """
        logs = {}
        for i, t in enumerate(self.tasks):
            wandb_obj = t.log_result(results[i],  dataset_obj.datasets[i], mode)
            logs[t.cfg.name + f'_{mode}'] = wandb_obj

        return logs

