from base import BaseDataLoader
from data_loader import MilabDataset

class MilabDataLoader(BaseDataLoader):
    def __init__ (self, data_dir, file_name, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = MilabDataset(data_dir + file_name)
        print(f"MilabDataset's loading compelete, dataset's length is {self.dataset.__len__()}")
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)