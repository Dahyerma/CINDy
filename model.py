import torch
import torch.nn as nn


class ParameterizedSINDy(nn.Module):
    def __init__(self, input_dim, output_dim, x_dim, lasso_regularization=0.0, hidden_dim=64):
        """
        :param input_dim: 입력 차원, Condition으로 주어지는 값의 갯수
        :param output_dim: 출력 차원, SINDY Matrix의 차원 (F x D)
        :param hidden_dim: 은닉층 차원, 기본값은 64
        """
        super(ParameterizedSINDy, self).__init__()
        self.x_dim = x_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.lasso_regularization = lasso_regularization

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def get_loss(self, X, next_X, condition):
        batch_size = X.size(0)
        dX = next_X - X
        predicted_SINDy = self.forward(condition).view(batch_size, -1, self.x_dim)
        predicted_differential = torch.einsum('bij,bjk->bik', predicted_SINDy, X.unsqueeze(-1)).squeeze(-1)
        
        loss = torch.mean((dX - predicted_differential) ** 2)
        if self.lasso_regularization > 0:
            lasso_loss = self.lasso_regularization * torch.sum(torch.abs(predicted_SINDy))
            loss += lasso_loss

        return loss

    def predict(self, X):
        return self.forward(X)