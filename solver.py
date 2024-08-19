import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_out)
        self.bn0 = nn.BatchNorm1d(n_in)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.bn0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class BSDEModelBS(nn.Module):
    """
    network for the example of geometric basket call option, max-call option
    """
    def __init__(self, cfg, option, k, init_c=None, init_grad=None):
        super(BSDEModelBS, self).__init__()
        self.option = option
        self.dt = option.dt
        self.hidden_dim = cfg['hidden_dim']
        self.vol = cfg['vol']
        self.num_time_step = cfg['num_time_step']
        self.d = cfg['d']
        self.k = k
        self.c_network = MLP(self.d + 1, 1, self.hidden_dim)  # input is [payoff, s]
        self.grad_network = MLP(self.d, self.d, self.hidden_dim)
        if init_c is not None:
            self.c_network.load_state_dict(init_c.state_dict(), strict=False)
        else:
            nn.init.xavier_uniform_(self.c_network.fc1.weight)
            nn.init.xavier_uniform_(self.c_network.fc2.weight)
            nn.init.xavier_uniform_(self.c_network.fc3.weight)
        if init_grad is not None:
            self.grad_network.load_state_dict(init_grad.state_dict(), strict=False)
        else:
            nn.init.xavier_uniform_(self.grad_network.fc1.weight)
            nn.init.xavier_uniform_(self.grad_network.fc2.weight)
            nn.init.xavier_uniform_(self.grad_network.fc3.weight)

    def forward(self, ss, dw, y_in, tau):
        input_size = ss.shape[0]
        z = self.vol * torch.mul(ss[:, 1:], dw[:, :])               # \sigma(S_k) dW_k
        g = self.grad_network(ss[:, 1:])
        y_k = torch.sum(torch.mul(g, z), dim=1, keepdim=True)
        if self.k == self.num_time_step - 1:
            y_out = y_k
            cv_out = self.c_network(ss)
            output = cv_out + y_out
        else:
            y_out = torch.cat([y_k, y_in], dim=1)
            ans = torch.cumsum(y_out, dim=1)
            y_sum = ans[range(input_size), tau - 1 - self.k].reshape((input_size, 1))
            cv_out = self.c_network(ss)
            output = cv_out + y_sum
        return output, cv_out, y_out


class BSDEModelHeston(nn.Module):
    """
    network for Heston model
    """
    def __init__(self, cfg, option, k, init_c=None, init_grad=None):
        super(BSDEModelHeston, self).__init__()
        self.option = option
        self.dt = option.dt
        self.hidden_dim = cfg['hidden_dim']
        self.cor = cfg['correlation']
        self.sqrt_cor = torch.sqrt(torch.tensor(1 - self.cor ** 2))
        self.nu = cfg['nu']
        self.vol_mat_T = torch.tensor([[self.cor, self.nu], [self.sqrt_cor, 0.0]])
        self.num_time_step = cfg['num_time_step']
        self.d = cfg['d']
        self.k = k
        self.c_network = MLP(self.d + 1, 1, self.hidden_dim)  # input is [phi, x, v]
        self.grad_network = MLP(self.d, self.d, self.hidden_dim)
        if init_c is not None:
            self.c_network.load_state_dict(init_c.state_dict(), strict=False)
        else:
            nn.init.xavier_uniform_(self.c_network.fc1.weight)
            nn.init.xavier_uniform_(self.c_network.fc2.weight)
            nn.init.xavier_uniform_(self.c_network.fc3.weight)
        if init_grad is not None:
            self.grad_network.load_state_dict(init_grad.state_dict(), strict=False)
        else:
            nn.init.xavier_uniform_(self.grad_network.fc1.weight)
            nn.init.xavier_uniform_(self.grad_network.fc2.weight)
            nn.init.xavier_uniform_(self.grad_network.fc3.weight)

    def forward(self, xx, dw, y_in, tau):
        input_size = xx.shape[0]
        z = torch.sqrt(xx[:, 2]).reshape((input_size, 1)) * (dw[:, :] @ self.vol_mat_T)  # \sigma(X_k) dW_k
        g = self.grad_network(xx[:, 1:])
        y_k = torch.sum(torch.mul(g, z), dim=1, keepdim=True)
        if self.k == self.num_time_step - 1:
            y_out = y_k
            cv_out = self.c_network(xx)
            output = cv_out + y_out
        else:
            y_out = torch.cat([y_k, y_in], dim=1)
            ans = torch.cumsum(y_out, dim=1)
            y_sum = ans[range(input_size), tau - 1 - self.k].reshape((input_size, 1))
            cv_out = self.c_network(xx)
            output = cv_out + y_sum
        return output, cv_out, y_out


class BSDEModelStrangle(nn.Module):
    """
    network for strangle spread basket option
    """
    def __init__(self, cfg, option, k, init_c=None, init_grad=None):
        super(BSDEModelStrangle, self).__init__()
        self.option = option
        self.dt = option.dt
        self.hidden_dim = cfg['hidden_dim']
        self.vol = torch.tensor([[0.3024,   0.1354,   0.0722,   0.1367,   0.1641],
                                 [0.1354,   0.2270,   0.0613,   0.1264,   0.1610],
                                 [0.0722,   0.0613,   0.0717,   0.0884,   0.0699],
                                 [0.1367,   0.1264,   0.0884,   0.2937,   0.1394],
                                 [0.1641,   0.1610,   0.0699,   0.1394,   0.2535]])
        self.num_time_step = cfg['num_time_step']
        self.d = cfg['d']
        self.k = k
        self.c_network = MLP(self.d + 1, 1, self.hidden_dim)  # input is [phi(s), s]
        self.grad_network = MLP(self.d, self.d, self.hidden_dim)
        if init_c is not None:
            self.c_network.load_state_dict(init_c.state_dict(), strict=False)
        else:
            nn.init.xavier_uniform_(self.c_network.fc1.weight)
            nn.init.xavier_uniform_(self.c_network.fc2.weight)
            nn.init.xavier_uniform_(self.c_network.fc3.weight)
        if init_grad is not None:
            self.grad_network.load_state_dict(init_grad.state_dict(), strict=False)
        else:
            nn.init.xavier_uniform_(self.grad_network.fc1.weight)
            nn.init.xavier_uniform_(self.grad_network.fc2.weight)
            nn.init.xavier_uniform_(self.grad_network.fc3.weight)

    def forward(self, ss, dw, y_in, tau):
        input_size = ss.shape[0]
        z = torch.mul(ss[:, 1:], dw[:, :] @ self.vol.T)             # \sigma(S_k) dW_k
        g = self.grad_network(ss[:, 1:])
        y_k = torch.sum(torch.mul(g, z), dim=1).reshape((input_size, 1))
        if self.k == self.num_time_step - 1:
            y_out = y_k
            cv_out = self.c_network(ss)
            output = cv_out + y_out
        else:
            y_out = torch.cat([y_k, y_in], dim=1)
            ans = torch.cumsum(y_out, dim=1)
            y_sum = ans[range(input_size), tau - 1 - self.k].reshape((input_size, 1))
            cv_out = self.c_network(ss)
            output = cv_out + y_sum
        return output, cv_out, y_out
