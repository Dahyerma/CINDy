import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
import yaml

from data_util import PEMFCDataset
from model import ParameterizedSINDy

# ------------------------------------------------------------------------------
# 1. 데이터 불러오기 · 전처리 (스무딩 포함)
# ------------------------------------------------------------------------------
from statsmodels.nonparametric.smoothers_lowess import lowess

def smooth_voltage(x, y, frac=0.2):
    """
    LOESS 스무딩 후, 원래 x 지점에 대응하는 y_smooth 반환
    x, y: 1D numpy arrays
    """
    sm = lowess(y, x, frac=frac, return_sorted=True)
    xs, ys = sm[:,0], sm[:,1]
    return np.interp(x, xs, ys)

all_dfs = []
for fp in glob.glob('data/RHa_*_RHc_*_*.csv'):
    df0 = pd.read_csv(fp, encoding='cp949')
    if 'Voltage(V)' not in df0.columns or 'Current(A)' not in df0.columns:
        continue

    # 전압·전류 컬럼 정리
    df = (
        df0[['Voltage(V)', 'Current(A)']]
        .dropna()
        .rename(columns={'Voltage(V)': 'Voltage'})
    )
    # 전류밀도 계산
    df['CurrentDensity'] = df['Current(A)'] / 25.0

    # LOESS 스무딩 적용
    df['Voltage'] = smooth_voltage(
        df['CurrentDensity'].values,
        df['Voltage'].values,
        frac=0.2
    )

    # 파일명에서 RH_a, RH_c, cycle 파싱
    m = re.search(r'RHa_(\d+)_RHc_(\d+)_(\d+)\.csv$', fp)
    if not m:
        continue
    df['RH_a']      = int(m.group(1))
    df['RH_c']      = int(m.group(2))
    df['cycle']     = int(m.group(3))
    # 0–1 정규화
    df['RH_a_norm'] = df['RH_a'] / 100.0
    df['RH_c_norm'] = df['RH_c'] / 100.0

    all_dfs.append(df[['Voltage','CurrentDensity','RH_a_norm','RH_c_norm','cycle']])

if not all_dfs:
    raise RuntimeError("데이터가 하나도 읽히지 않았습니다. 파일 패턴과 컬럼명을 확인하세요.")
df_all = pd.concat(all_dfs, ignore_index=True)

# ------------------------------------------------------------------------------
# 2. dV/dn 계산
# ------------------------------------------------------------------------------
df_all['dV_dn'] = (
    df_all
    .groupby(['CurrentDensity','RH_a_norm','RH_c_norm'])['Voltage']
    .diff()
)
df = df_all.dropna(subset=['dV_dn']).reset_index(drop=True)




config = yaml.safe_load(open('config.yaml', 'r'))
breakpoint()


dataset = PEMFCDataset(df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

epochs = 1000
model = ParameterizedSINDy(input_dim=3, output_dim=4, x_dim=2)  # 3개의 입력 (RH_a, RH_c, cycle), 4개의 출력 (dV_dn)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   
loss_fn = nn.MSELoss()

for epoch in trange(epochs, desc='Training'):
    for x, next_X, condition in dataloader:
        optimizer.zero_grad()
        loss = model.get_loss(x, next_X, condition)  # Voltage는 X[:, 0:1]
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


