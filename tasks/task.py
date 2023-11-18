import yaml
import abc
from ml_collections import config_dict
import util
import registry


TaskRegistry = registry.Registry()

class Task(abc.ABC):
    """Task class.
    Providing:
    - Preprocessing functions for a specific task that turns raw features into
    inputs in a common interface.
    - Post-processing functions for a specific task that decode the model's
     outputs in a common interface.
    - Evaluation for a specific task.
    - Important task properties, such as vocab size, max seq len.
    """

    def __init__(self, config : config_dict.ConfigDict):
       ## TODO can take in task config
       #with open(config_path, 'r') as f:
       #    self.cfg = config_dict.ConfigDict(yaml.safe_load(f))
       self.cfg = config

    @property
    def task_vocab_id(self):
        return self.cfg.vocab_id

    @abc.abstractmethod
    def preprocess_target(self, batch_targets):
        """
        Task-specific preprocessing of raw targets. For example detection targets will be in
        {'bbox': [[...]] , 'labels':[...]} for each example. We will convert it to batched_example
        by padding it to same dimensions.

        Args:
            batch_targets : List[Dict]
        """



 
    @abc.abstractmethod
    def preprocess_batched(self, batched_examples, training):
        """Task-specific preprocessing of batched examples on accelerators (TPUs).
            Args:
                  batched_examples: preprocessed and batched examples.
                  training: bool.
            Returns batched inputs in a comon interface for modeling.
         """

    @abc.abstractmethod
    def postprocess(self,im, target, logits, dataset_obj, **kwargs):
        """
        Task-specific postprocess such as draw image
        All tasks (including multitask) will take in arguments in the following order
            im , target, logits, dataset_obj

        target: for object detection task, target is boxes. For instance segmentation, target is mask.
        """



