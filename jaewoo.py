# train.py  ────────────────────────────────────────────────────────────────────
import glob, re, numpy as np, pandas as pd, yaml, torch
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from torch.utils.data import DataLoader
from data_util import PEMFCDataset
from model import ParameterizedSINDy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def smooth_voltage(x, y, frac=0.2):
    sm = lowess(y, x, frac=frac, return_sorted=True)
    xs, ys = sm[:,0], sm[:,1]
    return np.interp(x, xs, ys)

# ── (A) 원시 데이터 적재
all_dfs = []
for fp in glob.glob('data/RHa_*_RHc_*_*.csv'):
    df0 = pd.read_csv(fp, encoding='cp949')
    if 'Voltage(V)' not in df0.columns or 'Current(A)' not in df0.columns:
        continue
    df = (df0[['Voltage(V)', 'Current(A)']]
          .dropna()
          .rename(columns={'Voltage(V)':'Voltage'}))
    df['CurrentDensity'] = df['Current(A)']/25.0
    df['Voltage'] = smooth_voltage(df['CurrentDensity'].values, df['Voltage'].values, frac=0.2)

    m = re.search(r'RHa_(\d+)_RHc_(\d+)_(\d+)\.csv$', fp)
    if not m: 
        continue
    df['RH_a'] = int(m.group(1)); df['RH_c'] = int(m.group(2)); df['cycle'] = int(m.group(3))
    df['RH_a_norm'] = df['RH_a']/100.0; df['RH_c_norm'] = df['RH_c']/100.0
    all_dfs.append(df[['Voltage','CurrentDensity','RH_a_norm','RH_c_norm','cycle']])

if not all_dfs:
    raise RuntimeError("데이터가 하나도 읽히지 않았습니다.")

df_all = pd.concat(all_dfs, ignore_index=True)
cycle_max = df_all['cycle'].max()
df_all['cycle_norm'] = df_all['cycle'] / max(1, cycle_max)

# ── (B) n→n+1 페어 만들기 (같은 RH, 같은 V_bin에서)
df_all = df_all.sort_values(['RH_a_norm','RH_c_norm','Voltage','cycle']).reset_index(drop=True)

# 1) 전압 binning (필요시 자릿수 조절: 3~5)
df_all['V_bin'] = df_all['Voltage'].round(4)

# 2) 왼쪽: cycle = n
left = df_all[['RH_a_norm','RH_c_norm','V_bin','cycle','cycle_norm','Voltage','CurrentDensity']].rename(
    columns={'CurrentDensity': 'I_n'}
)

# 3) 오른쪽: cycle = n+1 을 n에 맞추도록 cycle을 -1 이동
right = df_all[['RH_a_norm','RH_c_norm','V_bin','cycle','CurrentDensity']].rename(
    columns={'CurrentDensity': 'I_np1', 'cycle': 'cycle_next'}
)
right['cycle'] = right['cycle_next'] - 1
right = right.drop(columns=['cycle_next'])

# 4) 키 조인: (RH, V_bin, cycle) 기준으로 n ↔ n+1 매칭
df_pairs = pd.merge(
    left, right,
    on=['RH_a_norm','RH_c_norm','V_bin','cycle'],
    how='inner'
)

# 5) Dataset이 기대하는 컬럼명으로 정리
df_pairs = df_pairs.rename(columns={'I_np1': 'NextCurrentDensity',
                                    'I_n':   'CurrentDensity'})

# 최종 컬럼만 남기기
df_pairs = df_pairs[['Voltage','CurrentDensity','NextCurrentDensity',
                     'RH_a_norm','RH_c_norm','cycle','cycle_norm']]


# ── (C) Dataset/Loader
config = yaml.safe_load(open('./config/config.yaml', 'r'))
dataset = PEMFCDataset(df_pairs)
dataset.preprocess(config)             # Θ(V,I_n)
F = dataset.theta.shape[1]             # 라이브러리 항 수

loader = DataLoader(dataset, batch_size=1024, shuffle=True, pin_memory=(device.type=='cuda'))

# ── (D) Model/Train (I만)
model = ParameterizedSINDy(
    input_dim=3,               # RH_a_norm, RH_c_norm, cycle_norm
    output_dim=F,              # F * D (D=1)
    x_dim=1,
    lasso_regularization=1e-4,
    hidden_dim=64
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1000):
    for x, next_X, cond, th in loader:

        x, next_X, cond, th = x.to(device), next_X.to(device), cond.to(device), th.to(device)
        opt.zero_grad()
        loss = model.get_loss(x, next_X, th, cond)
        loss.backward()
        opt.step()
    if (epoch+1) % 100 == 0:
        print(f"[{epoch+1}] loss={loss.item():.6f}")

# ── (E) 추론: Θ·Ξ로 dI 누적 (V 그리드 고정)
def compute_dI_step(model, V_np, I_np, cond_np, cfg):
    theta_np, _ = dataset.make_library(V_np.astype(np.float32), I_np.astype(np.float32), cfg)
    with torch.no_grad():
        theta  = torch.tensor(theta_np, dtype=torch.float32, device=device) # [B,F]
        cond   = torch.tensor(cond_np,  dtype=torch.float32, device=device) # [B,3]
        Xi     = model.forward(cond)                                        # [B,F]
        dI     = torch.sum(Xi * theta, dim=1)                               # [B]
        return dI.cpu().numpy()

# 예시: RH=30/80, cycle 0→15
c1, c2 = 0.30, 0.80
target_cycle = 15

df_cond = (df_all[(df_all['RH_a_norm']==c1) & (df_all['RH_c_norm']==c2)]
           .sort_values(['cycle','Voltage']).reset_index(drop=True))
df_init = df_cond[df_cond['cycle']==0]
V_grid  = df_init['Voltage'].values.astype(np.float32).copy()
I_pred  = df_init['CurrentDensity'].values.astype(np.float32).copy()

for n in range(0, target_cycle):
    cond_batch = np.column_stack([
        np.full_like(V_grid, c1, dtype=np.float32),
        np.full_like(V_grid, c2, dtype=np.float32),
        np.full_like(V_grid, (n+1)/max(1, cycle_max), dtype=np.float32)   # cycle_norm
    ])
    dI = compute_dI_step(model, V_grid, I_pred, cond_batch, config)
    I_pred = I_pred + dI

# 시각화 (선택)
import matplotlib.pyplot as plt
def smooth_voltage_curve(V, I, frac=0.2):
    return smooth_voltage(V, I, frac=frac)

I_pred_smooth = smooth_voltage_curve(V_grid, I_pred, frac=0.2)
df_tgt = df_cond[df_cond['cycle']==target_cycle]

plt.figure(figsize=(8,6))
plt.plot(df_tgt['CurrentDensity'], df_tgt['Voltage'], 'o', label=f'Actual Cycle {target_cycle}')
plt.plot(I_pred_smooth, V_grid, '-', label=f'Predicted (smoothed) Cycle {target_cycle}')
plt.xlabel('Current Density (A/cm²)'); plt.ylabel('Voltage (V)')
plt.title(f'IV Curve @ RH={int(c1*100)}/{int(c2*100)}, Cycle {target_cycle}')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig('iv_curve_prediction_I_Re.png')
