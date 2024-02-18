import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# class MarineFoulingDataset(Dataset):
#     def __init__(self, X, y):
#         # Ensure X is a dataframe and not just numpy array to use .apply
#         if isinstance(X, pd.DataFrame):
#             X = X.apply(pd.to_numeric, errors='coerce').fillna(0).values
#         else:
#             X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(0).values
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.long)
        
#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

class MarineFoulingDataset(Dataset):
    def __init__(self, X, y):
        X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(0).values
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

