
# dataloader for box detection task and image level classification task
# Detection : BHX (Subset of CQ500)
# Classification: CQ500

from data import dataset as dataset_lib


#@dataset_lib.DatasetRegistry.register('multitask')
class MultiDataset():
    def __init__(self,datasets):
        '''
        Args:
            datasets: `List` dataset_lib.Dataset
        '''
        self.datasets = datasets


    def __getitem__(self, index):
        results = []
        for d in self.datasets:
            results += [d.__getitem__(index)]
        return results


    def load_dataloader(self,mode ):
        dataloaders = []
        for d in self.datasets:
            dataloaders += [d.load_dataloader(mode)]
        return dataloaders

    def process(self,batch, idx=-1):
        if idx >= 0:
            return self.datasets[idx].process(batch)
        results = []
        for i in range(len(self.datasets)):
            results += [self.datasets[i].process(batch[i])]

        return results






