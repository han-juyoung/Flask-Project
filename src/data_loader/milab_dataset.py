import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class MilabDataset(Dataset):
 def __init__ (self, file_name):
  df = pd.read_excel(file_name)
  columns_len = len(df.columns)

  df_x = df.iloc[:, 1:columns_len].values
  df_y = np.reshape(df.iloc[:, 0].values, (-1, 1))

  self.x=torch.tensor(df_x, dtype=torch.float32)
  self.y=torch.tensor(df_y, dtype=torch.float32)

 def __len__(self):
  return len(self.y)

 def __getitem__(self, idx):
  return self.x[idx], self.y[idx]
  

if __name__ == '__main__':
    data_train = MilabDataset('data/train_set_pre_worked.xlsx')
    data_test = MilabDataset('data/test_set_pre_worked.xlsx')
    print(data_train.y.size())
    print(data_test.y.size())
    