# model.py  ────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class ParameterizedSINDy(nn.Module):
    def __init__(self, input_dim, output_dim, x_dim=1, lasso_regularization=0.0, hidden_dim=64):
        """
        input_dim: φ 차원 (RH_a_norm, RH_c_norm, cycle_norm → 3)
        output_dim: F * D (여기선 D=1 → F)
        x_dim: 상태 차원 (I만 학습 → 1)
        """
        super().__init__()
        self.x_dim = x_dim  # 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.lasso_regularization = lasso_regularization

    def forward(self, cond):  # cond = φ
        h = self.activation(self.fc1(cond))
        return self.fc2(h)  # [B, F]

    def get_loss(self, X, next_X, theta, condition):
        """
        Θ(V,I_n)·Ξ(φ) 로 dI 예측
        X, next_X: [..., 2] (V,I)
        theta: [B, F]
        condition: [B, C]
        """
        I   = X[:, 1]           # I_n
        I_p = next_X[:, 1]      # I_{n+1}
        dI  = I_p - I           # [B]

        B, F = theta.size(0), theta.size(1)
        Xi   = self.forward(condition).view(B, F)     # Ξ(φ): [B, F]
        pred = torch.sum(Xi * theta, dim=1)           # Θ·Ξ → [B]

        loss = torch.mean((dI - pred) ** 2)
        if self.lasso_regularization > 0:
            loss += self.lasso_regularization * Xi.abs().mean()
        return loss
