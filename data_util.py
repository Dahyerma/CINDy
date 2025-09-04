# data_util.py  ────────────────────────────────────────────────────────────────
import torch
from torch.utils.data import Dataset
import numpy as np

class PEMFCDataset(Dataset):
    def __init__(self, df, target_cycle=None, preprocess_config=None):
        """
        df: 반드시 다음 컬럼을 포함
            ['Voltage','CurrentDensity','NextCurrentDensity',
             'RH_a_norm','RH_c_norm','cycle','cycle_norm']
        """
        if target_cycle is not None:
            df = df[df['cycle'] == target_cycle].reset_index(drop=True)

        # X = [V, I_n], next_X = [V, I_{n+1}]
        self.X       = df[['Voltage', 'CurrentDensity']].values.astype(np.float32)
        self.next_X  = np.column_stack([
            df['Voltage'].values, df['NextCurrentDensity'].values
        ]).astype(np.float32)

        # φ = [RH_a_norm, RH_c_norm, cycle_norm]
        self.condition = df[['RH_a_norm','RH_c_norm','cycle_norm']].values.astype(np.float32)
        self.cycles    = df['cycle'].values.astype(np.int32)

        self.theta = None      # Θ(V,I)
        self.theta_names = []
        self.cfg = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x0        = torch.tensor(self.X[idx], dtype=torch.float32)        # [V,I]
        x1        = torch.tensor(self.next_X[idx], dtype=torch.float32)   # [V,I_next]
        condition = torch.tensor(self.condition[idx], dtype=torch.float32)
        th        = torch.tensor(self.theta[idx], dtype=torch.float32)    # [F]
        return x0, x1, condition, th

    # ── SINDy 라이브러리 Θ(V,I)
    def make_library(self, V, I, cfg):
        order   = cfg.get('polynomial_order', 1)
        add_trig = cfg.get('trigonometric_function', False)   # ← 오타 수정
        add_exp  = cfg.get('exponential_function', False)
        clip     = cfg.get('exp_arg_clip', 8.0)

        feat_cols, feat_names = [], []

        # 1) Polynomial terms (total degree = 1..order)
        for total_deg in range(1, order + 1):
            for a in range(total_deg + 1):
                b = total_deg - a
                feat_cols.append((V ** a) * (I ** b))
                feat_names.append(f'V^{a}_I^{b}')

        # 2) Trig
        if add_trig:
            feat_cols += [np.sin(I).astype(np.float32), np.cos(I).astype(np.float32)]
            feat_names += ['sin(I)', 'cos(I)']

        # 3) Exp
        if add_exp:
            I_clip = np.clip(I, -clip, clip)
            feat_cols += [np.exp(I_clip).astype(np.float32)]
            feat_names += ['exp(I)']

        theta = np.column_stack(feat_cols).astype(np.float32)
        return theta, feat_names

    def preprocess(self, cfg: dict):
        self.cfg = cfg
        V = self.X[:, 0]
        I = self.X[:, 1]                  # 현재 사이클의 I_n
        theta, names = self.make_library(V, I, cfg)
        self.theta = theta                # Θ(V, I_n)
        self.theta_names = names
