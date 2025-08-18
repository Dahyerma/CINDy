import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class PEMFCDataset(Dataset):
    def __init__(self, df, target_cycle=None, preprocess_config=None):
        """
        PEMFC 데이터셋 초기화
        :param df: DataFrame with columns ['Voltage', 'CurrentDensity', 'RH_a', 'RH_c', 'cycle']
        :param target_cycle: 특정 사이클에 대한 데이터만 사용할 경우 지정
        :param preprocess_config: 사전 처리 설정 (예: 정규화, 추가 피처 생성 등) --> key: "power", 
        """
        if target_cycle is not None:
            df = df[df['cycle'] == target_cycle].reset_index(drop=True)
        
        self.X = df[['Voltage', 'CurrentDensity']].values
        self.condition = df[['RH_a_norm', 'RH_c_norm', 'cycle']].values

        self.ns = df['Ns'].values

    def __len__(self):
        return len(self.X) - 1

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.X[idx + 1], dtype=torch.float32), torch.tensor(self.condition[idx], dtype=torch.float32)
    
    ## TODO tasks ##
    ## 1. 데이터셋을 사이클 별로도 쪼개야 합니다. 
    ## 2. I, V 외에도 I^2, Sin(I), Cos(I), I*V I^2*V ,.... 등의 갖가지 변술르 X에 포함시키세요.

    def divide_cycle(self, whole_sequence):
        ns = np.array(self.ns)
        n = len(ns)
        if n == 0:
            return [] if whole_sequence else {}

        cycles = []
        start_idx = 0
        for i in range(1, n):
            if ns[i-1] == 2 and ns[i] == 0:
                cycles.append((start_idx, i))
                start_idx = i
        
        cycles.append((start_idx, n))

        if whole_sequence:
            return cycles
        
        return {idx: (s, e) for idx, (s, e) in enumerate(cycles)}
        
    
    def preprocess(self):
        # self.X[:, -1] = self.X[0] ** 2 
        raise NotImplementedError("Preprocessing method not implemented. Please implement the preprocessing logic as needed.")