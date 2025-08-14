import torch
from torch.utils.data import Dataset, DataLoader


class PEMFCDataset(Dataset):
    def __init__(self, df, target_cycle=None):
        """
        PEMFC 데이터셋 초기화
        :param df: DataFrame with columns ['Voltage', 'CurrentDensity', 'RH_a', 'RH_c', 'cycle']
        :param target_cycle: 특정 사이클에 대한 데이터만 사용할 경우 지정
        """
        if target_cycle is not None:
            df = df[df['cycle'] == target_cycle].reset_index(drop=True)
        
        self.X = df[['Voltage', 'CurrentDensity']].values
        self.condition = df[['RH_a_norm', 'RH_c_norm', 'cycle']].values

    def __len__(self):
        return len(self.X) - 1

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.X[idx + 1], dtype=torch.float32), torch.tensor(self.condition[idx], dtype=torch.float32)