
import abc
import functools
from typing import Callable
from ml_collections import config_dict
import registry


DatasetRegistry = registry.Registry()


class Dataset(abc.ABC):
  """A dataset that handles creating a tf.data.Dataset."""

  def __init__(self, config: config_dict.ConfigDict):
    """Constructs the dataset."""
    self.config = config.dataset
    self.task_config = config.task


  @abc.abstractmethod
  def load_dataloader(self, mode: str):
      """
      Args:
          mode : 'train', 'val' or 'test'
      Return:
          torch dataloader object for to be used in lightning.
      """
  def process(self, batch, training=True):
      """
      To process targets (for ex: detection to be padded). If no required preprocess
      then return identity.

      Args:
          batch : (Tuple) Dataloader will return (image, target) in lightning train step.

      Return:
          (im, target) to task.preprocess_batched

      """

      return batch
      



