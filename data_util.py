import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class PEMFCDataset(Dataset):
    def __init__(self, df, target_cycle=None, preprocess_config=None):
        """
        PEMFC 데이터셋 초기화
        :param df: DataFrame with columns ['Voltage', 'CurrentDensity', 'RH_a_norm', 'RH_c_norm', 'cycle']
        :param target_cycle: 특정 사이클에 대한 데이터만 사용할 경우 지정
        :param preprocess_config: 사전 처리 설정 (예: 정규화, 추가 피처 생성 등) --> key: "power", 
        """
        if target_cycle is not None:
            df = df[df['cycle'] == target_cycle].reset_index(drop=True)
        
        self.X = df[['Voltage', 'CurrentDensity']].values
        self.condition = df[['RH_a_norm', 'RH_c_norm', 'cycle']].values
        # 사이클 구분은 'Ns' 대신 파일명에서 파싱한 'cycle' 값을 사용
        self.cycles = df['cycle'].values

        self.theta = None
        self.theta_names = []
        self.cfg = None
      

    def __len__(self):
        return max(0, len(self.X) - 1)

    def __getitem__(self, idx):
        x0=torch.tensor(self.X[idx], dtype=torch.float32)
        x1=torch.tensor(self.X[idx + 1], dtype=torch.float32)
        condition = torch.tensor(self.condition[idx], dtype=torch.float32)
        th = torch.tensor(self.theta[idx], dtype=torch.float32) 
        
        return x0, x1, condition, th
      
    ## TODO tasks ##
    ## 1. 데이터셋을 사이클 별로도 쪼개야 합니다. 
    ## 2. I, V 외에도 I^2, Sin(I), Cos(I), I*V I^2*V ,.... 등의 갖가지 변수를 X에 포함시키세요.

    def divide_cycle(self, whole_sequence):
        cycles_arr = np.array(self.cycles)
        global start
        n = len(cycles_arr)
        if n == 0:
            return [] if whole_sequence else {}

        ranges = []
        start = 0
        for i in range(1, n):
            if cycles_arr[i] != cycles_arr[i-1]:
                ranges.append((start, i))
                start = i
        ranges.append((start, n))


        if whole_sequence:
            out = []
            for start, end in ranges:
                if end - start >= 2:
                    out.append(np.arange(start, end - 1))
                else:
                    out.append([])
            return out
        else:
            result = {}
            for start, end in ranges:
                c = cycles_arr[start]
                if end - start >= 2:
                    result[c] = np.arange(start, end - 1)
                else:
                    result[c] = []
            return result
        
    def make_library(self, V, I, cfg):
        order = (cfg.get('polynomial_order', 1))
        add_trig = cfg.get('triangular_function', False)
        add_exp = cfg.get('exponential_function', False)
        clip = cfg.get('exp_arg_clip', 8.0)

        feat_cols = []
        feat_names = []

        #1) Polynomial features
        for total_deg in range(1, order + 1):
            for a in range(total_deg + 1):
                b = total_deg - a
                feat_cols.append((V ** a) * (I ** b))
                feat_names.append(f'V^{a}_I^{b}')

        #2) Trigonometric features
        if add_trig:
            feat_cols +=[np.sin(I).astype(np.float32), np.cos(I).astype(np.float32)]
            feat_names += ['sin(I)', 'cos(I)']

        #3) Exponential features
        if add_exp:
            I_clip = np.clip(I, -clip, clip)
            feat_cols += [np.exp(I_clip).astype(np.float32)]
            feat_names += ['exp(I)']

        theta = np.column_stack(feat_cols).astype(np.float32)
        return theta, feat_names


            
    def preprocess(self, cfg: dict):
        self.cfg = cfg
        V = self.X[:, 0]
        I = self.X[:, 1]
        theta, names = self.make_library(V, I, cfg)
        self.theta = theta
        self.theta_names = names
