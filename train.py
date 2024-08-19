import torch
import torch.nn as nn
from solver import BSDEModelBS, BSDEModelHeston, BSDEModelStrangle
import time
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR


class DeepBSDEOSGeoBaskCall:
    """
    add phi feature to c_fun, where payoff = max(phi, 0)
    grad_fun as a function of s^1, ..., s^d
    """
    def __init__(self, cfg, option):
        self.cfg = cfg
        self.batch_size = cfg['batch_size']
        self.num_iterations = cfg['num_iterations']
        self.num_epochs = cfg['num_epochs']
        self.num_time_step = cfg['num_time_step']
        self.d = cfg['d']
        self.logging_frequency = cfg['logging_frequency']
        self.hidden_dim = cfg['hidden_dim']
        self.lr = cfg['lr']
        self.vol = cfg['vol']
        self.option = option
        self.device = torch.device(cfg['device'])

    def train(self):
        c_fun = {}
        g_fun = {}
        num_samples = self.batch_size * self.num_iterations
        dw_train, s_train = self.option.sample(num_samples)
        phi_matrix = self.option.phi(s_train)
        y_mat = torch.zeros((num_samples, 1), device=self.device)
        tau = torch.ones(num_samples, dtype=torch.int32, device=self.device) * self.num_time_step

        dw_train = dw_train.to(self.device)
        s_train = s_train.to(self.device)
        phi_matrix = phi_matrix.to(self.device)
        payout = torch.maximum(phi_matrix[:, self.num_time_step], torch.tensor([0.0]))

        print("start training!")
        start_time = time.time()
        for k in range(self.num_time_step - 1, 0, -1):
            if k == self.num_time_step - 1:
                model = BSDEModelBS(self.cfg, self.option, k)  # (1+d) input
            else:
                model = BSDEModelBS(self.cfg, self.option, k, init_c=c_fun[k+1], init_grad=g_fun[k+1])
            model.to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            if k == self.num_time_step - 1:
                scheduler = LambdaLR(optimizer, lr_lambda=lambda n: 1e-2 * 1e-4 ** (max((n - 50) / 1000, 0)))
                num_epochs = self.num_epochs * 2
            else:
                scheduler = LambdaLR(optimizer, lr_lambda=lambda n: 1e-2 * 1e-4 ** (max((n - 50) / 500, 0)))
                num_epochs = self.num_epochs

            model.train()
            i = 0
            for epoch in range(num_epochs):
                for step in range(self.num_iterations):
                    idx_batch = range(step * self.batch_size, (step + 1) * self.batch_size)
                    ss_train_batch_k = torch.cat([phi_matrix[idx_batch, k].reshape((self.batch_size, 1)), s_train[idx_batch, :, k]], dim=1)
                    dw_train_batch_k = dw_train[idx_batch, :, k]
                    y_mat_batch = y_mat[idx_batch, :]
                    tau_batch = tau[idx_batch]
                    payout_batch = payout[idx_batch]

                    out, _, _ = model(ss_train_batch_k, dw_train_batch_k, y_mat_batch, tau_batch)
                    out = out.reshape(self.batch_size, )
                    loss_fun = nn.MSELoss()
                    loss = loss_fun(out, payout_batch)
                    loss.to(self.device)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 50)
                    optimizer.step()
                    scheduler.step()

                    i = i + 1
                    if i % self.logging_frequency == 0:
                        print(f"step: {i}, loss: {loss.item():.4f}, time: {time.time() - start_time:.2f}, "
                              f"lr: {optimizer.param_groups[0]['lr']:.6f}")

            # update payout, y_mat
            with torch.no_grad():
                model.eval()
                ss_train_k = torch.cat([phi_matrix[:, k].reshape((num_samples, 1)), s_train[:, :, k]], dim=1)
                _, continue_value, y_mat = model(ss_train_k, dw_train[:, :, k], y_mat, tau)
                continue_value = continue_value.reshape(num_samples, )

                exercise_value = torch.maximum(phi_matrix[:, k], torch.tensor([0.0]))
                idx_exercise = (exercise_value >= continue_value) & (exercise_value > 0)
                payout[idx_exercise] = exercise_value[idx_exercise]
                tau[idx_exercise] = k

            c_fun.update({k: model.c_network})
            g_fun.update({k: model.grad_network})
            print(f"--------------The time step {k} is done!--------------")

        option_value = 1 / num_samples * torch.sum(payout)
        return c_fun, g_fun, option_value


class DeepBSDEOSStrangle:
    """
    add phi feature to c_fun, where payoff = max(phi, 0)
    grad_fun as a function of s^1, ..., s^d
    """
    def __init__(self, cfg, option):
        self.cfg = cfg
        self.batch_size = cfg['batch_size']
        self.num_iterations = cfg['num_iterations']
        self.num_epochs = cfg['num_epochs']
        self.num_time_step = cfg['num_time_step']
        self.d = cfg['d']
        self.logging_frequency = cfg['logging_frequency']
        self.hidden_dim = cfg['hidden_dim']
        self.lr = cfg['lr']
        self.option = option
        self.device = torch.device(cfg['device'])

    def train(self):
        c_fun = {}
        g_fun = {}
        num_samples = self.batch_size * self.num_iterations
        dw_train, s_train = self.option.sample(num_samples)
        phi_matrix = self.option.phi(s_train)
        y_mat = torch.zeros((num_samples, 1), device=self.device)
        tau = torch.ones(num_samples, dtype=torch.int32, device=self.device) * self.num_time_step

        dw_train = dw_train.to(self.device)
        s_train = s_train.to(self.device)
        phi_matrix = phi_matrix.to(self.device)
        payout = torch.maximum(phi_matrix[:, self.num_time_step], torch.tensor([0.0]))

        print("start training!")
        start_time = time.time()
        for k in range(self.num_time_step - 1, 0, -1):
            if k == self.num_time_step - 1:
                model = BSDEModelStrangle(self.cfg, self.option, k)  # (1+d) input
            else:
                model = BSDEModelStrangle(self.cfg, self.option, k, init_c=c_fun[k+1], init_grad=g_fun[k+1])
            model.to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            if k == self.num_time_step - 1:
                scheduler = LambdaLR(optimizer, lr_lambda=lambda n: 1e-2 * 1e-4 ** (max((n - 50) / 1000, 0)))
                num_epochs = self.num_epochs * 2
            else:
                scheduler = LambdaLR(optimizer, lr_lambda=lambda n: 1e-2 * 1e-4 ** (max((n - 50) / 500, 0)))
                num_epochs = self.num_epochs

            model.train()
            i = 0
            for epoch in range(num_epochs):
                for step in range(self.num_iterations):
                    idx_batch = range(step * self.batch_size, (step + 1) * self.batch_size)
                    ss_train_batch_k = torch.cat([phi_matrix[idx_batch, k].reshape((self.batch_size, 1)), s_train[idx_batch, :, k]], dim=1)
                    dw_train_batch_k = dw_train[idx_batch, :, k]
                    y_mat_batch = y_mat[idx_batch, :]
                    tau_batch = tau[idx_batch]
                    payout_batch = payout[idx_batch]

                    out, _, _ = model(ss_train_batch_k, dw_train_batch_k, y_mat_batch, tau_batch)
                    out = out.reshape(self.batch_size, )
                    loss_fun = nn.MSELoss()
                    loss = loss_fun(out, payout_batch)
                    loss.to(self.device)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 50)
                    optimizer.step()
                    scheduler.step()

                    i = i + 1
                    if i % self.logging_frequency == 0:
                        print(f"step: {i}, loss: {loss.item():.4f}, time: {time.time() - start_time:.2f}, "
                              f"lr: {optimizer.param_groups[0]['lr']:.6f}")

            # update payout, y_mat
            with torch.no_grad():
                model.eval()
                ss_train_k = torch.cat([phi_matrix[:, k].reshape((num_samples, 1)), s_train[:, :, k]], dim=1)
                _, continue_value, y_mat = model(ss_train_k, dw_train[:, :, k], y_mat, tau)
                continue_value = continue_value.reshape(num_samples, )

                exercise_value = torch.maximum(phi_matrix[:, k], torch.tensor([0.0]))
                idx_exercise = (exercise_value >= continue_value) & (exercise_value > 0)
                payout[idx_exercise] = exercise_value[idx_exercise]
                tau[idx_exercise] = k

            c_fun.update({k: model.c_network})
            g_fun.update({k: model.grad_network})
            print(f"--------------The time step {k} is done!--------------")

        option_value = 1 / num_samples * torch.sum(payout)
        return c_fun, g_fun, option_value


class DeepBSDEOSHeston:
    """
    add phi feature to c_fun, where payoff = max(phi, 0)
    grad_fun as a function of s^1, ..., s^d
    """
    def __init__(self, cfg, option):
        self.cfg = cfg
        self.batch_size = cfg['batch_size']
        self.num_iterations = cfg['num_iterations']
        self.num_epochs = cfg['num_epochs']
        self.num_time_step = cfg['num_time_step']
        self.d = cfg['d']
        self.logging_frequency = cfg['logging_frequency']
        self.hidden_dim = cfg['hidden_dim']
        self.lr = cfg['lr']
        self.option = option
        self.device = torch.device(cfg['device'])

    def train(self):
        c_fun = {}
        g_fun = {}
        num_samples = self.batch_size * self.num_iterations
        dw_train, xv_train = self.option.sample(num_samples)
        phi_matrix = self.option.phi(xv_train)
        y_mat = torch.zeros((num_samples, 1), device=self.device)
        tau = torch.ones(num_samples, dtype=torch.int32, device=self.device) * self.num_time_step

        dw_train = dw_train.to(self.device)
        xv_train = xv_train.to(self.device)
        phi_matrix = phi_matrix.to(self.device)
        payout = torch.maximum(phi_matrix[:, self.num_time_step], torch.tensor([0.0]))

        print("start training!")
        start_time = time.time()
        for k in range(self.num_time_step - 1, 0, -1):
            if k == self.num_time_step - 1:
                model = BSDEModelHeston(self.cfg, self.option, k)  # (1+d) input
            else:
                model = BSDEModelHeston(self.cfg, self.option, k, init_c=c_fun[k+1], init_grad=g_fun[k+1])
            model.to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            if k == self.num_time_step - 1:
                scheduler = LambdaLR(optimizer, lr_lambda=lambda n: 1e-2 * 1e-4 ** (max((n - 50) / 1000, 0)))
                num_epochs = self.num_epochs * 2
            else:
                scheduler = LambdaLR(optimizer, lr_lambda=lambda n: 1e-2 * 1e-4 ** (max((n - 50) / 500, 0)))
                num_epochs = self.num_epochs

            model.train()
            i = 0
            for epoch in range(num_epochs):
                for step in range(self.num_iterations):
                    idx_batch = range(step * self.batch_size, (step + 1) * self.batch_size)
                    xx_train_batch_k = torch.cat([phi_matrix[idx_batch, k].reshape((self.batch_size, 1)), xv_train[idx_batch, :, k]], dim=1)
                    dw_train_batch_k = dw_train[idx_batch, :, k]
                    y_mat_batch = y_mat[idx_batch, :]
                    tau_batch = tau[idx_batch]
                    payout_batch = payout[idx_batch]

                    out, _, _ = model(xx_train_batch_k, dw_train_batch_k, y_mat_batch, tau_batch)
                    out = out.reshape(self.batch_size, )
                    loss_fun = nn.MSELoss()
                    loss = loss_fun(out, payout_batch)
                    loss.to(self.device)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 50)
                    optimizer.step()
                    scheduler.step()

                    i = i + 1
                    if i % self.logging_frequency == 0:
                        print(f"step: {i}, loss: {loss.item():.4f}, time: {time.time() - start_time:.2f}, "
                              f"lr: {optimizer.param_groups[0]['lr']:.6f}")

            # update payout, y_mat
            with torch.no_grad():
                model.eval()
                xx_train_k = torch.cat([phi_matrix[:, k].reshape((num_samples, 1)), xv_train[:, :, k]], dim=1)
                _, continue_value, y_mat = model(xx_train_k, dw_train[:, :, k], y_mat, tau)
                continue_value = continue_value.reshape(num_samples, )

                exercise_value = torch.maximum(phi_matrix[:, k], torch.tensor([0.0]))
                idx_exercise = (exercise_value >= continue_value) & (exercise_value > 0)
                payout[idx_exercise] = exercise_value[idx_exercise]
                tau[idx_exercise] = k

            c_fun.update({k: model.c_network})
            g_fun.update({k: model.grad_network})
            print(f"--------------The time step {k} is done!--------------")

        option_value = 1 / num_samples * torch.sum(payout)
        return c_fun, g_fun, option_value


class DeepBSDEOSMaxCall:
    """
    add phi feature to c_fun, where payoff = max(phi, 0)
    grad_fun as a function of s^1, ..., s^d
    """
    def __init__(self, cfg, option):
        self.cfg = cfg
        self.batch_size = cfg['batch_size']
        self.num_iterations = cfg['num_iterations']
        self.num_epochs = cfg['num_epochs']
        self.num_time_step = cfg['num_time_step']
        self.d = cfg['d']
        self.logging_frequency = cfg['logging_frequency']
        self.hidden_dim = cfg['hidden_dim']
        self.lr = cfg['lr']
        self.vol = cfg['vol']
        self.option = option
        self.device = torch.device(cfg['device'])

    def train(self):
        c_fun = {}
        g_fun = {}
        num_samples = self.batch_size * self.num_iterations
        dw_train, s_train = self.option.sample(num_samples)
        phi_matrix = self.option.phi(s_train)
        y_mat = torch.zeros((num_samples, 1), device=self.device)
        tau = torch.ones(num_samples, dtype=torch.int32, device=self.device) * self.num_time_step

        dw_train = dw_train.to(self.device)
        s_train = s_train.to(self.device)
        phi_matrix = phi_matrix.to(self.device)
        payout = torch.maximum(phi_matrix[:, self.num_time_step], torch.tensor([0.0]))

        print("start training!")
        start_time = time.time()
        for k in range(self.num_time_step - 1, 0, -1):
            if k == self.num_time_step - 1:
                model = BSDEModelBS(self.cfg, self.option, k)  # (1+d) input
            else:
                model = BSDEModelBS(self.cfg, self.option, k, init_c=c_fun[k+1], init_grad=g_fun[k+1])
            model.to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            if k == self.num_time_step - 1:
                scheduler = LambdaLR(optimizer, lr_lambda=lambda n: 1e-2 * 1e-4 ** (max((n - 50) / 1000, 0)))
                num_epochs = self.num_epochs * 2
            else:
                scheduler = LambdaLR(optimizer, lr_lambda=lambda n: 1e-2 * 1e-4 ** (max((n - 50) / 500, 0)))
                num_epochs = self.num_epochs

            model.train()
            i = 0
            for epoch in range(num_epochs):
                for step in range(self.num_iterations):
                    idx_batch = range(step * self.batch_size, (step + 1) * self.batch_size)
                    ss_train_batch_k = torch.cat([phi_matrix[idx_batch, k].reshape((self.batch_size, 1)), s_train[idx_batch, :, k]], dim=1)
                    dw_train_batch_k = dw_train[idx_batch, :, k]
                    y_mat_batch = y_mat[idx_batch, :]
                    tau_batch = tau[idx_batch]
                    payout_batch = payout[idx_batch]

                    out, _, _ = model(ss_train_batch_k, dw_train_batch_k, y_mat_batch, tau_batch)
                    out = out.reshape(self.batch_size, )
                    loss_fun = nn.MSELoss()
                    loss = loss_fun(out, payout_batch)
                    loss.to(self.device)

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 50)
                    optimizer.step()
                    scheduler.step()

                    i = i + 1
                    if i % self.logging_frequency == 0:
                        print(f"step: {i}, loss: {loss.item():.4f}, time: {time.time() - start_time:.2f}, "
                              f"lr: {optimizer.param_groups[0]['lr']:.6f}")

            # update payout, y_mat
            with torch.no_grad():
                model.eval()
                ss_train_k = torch.cat([phi_matrix[:, k].reshape((num_samples, 1)), s_train[:, :, k]], dim=1)
                _, continue_value, y_mat = model(ss_train_k, dw_train[:, :, k], y_mat, tau)
                continue_value = continue_value.reshape(num_samples, )

                exercise_value = torch.maximum(phi_matrix[:, k], torch.tensor([0.0]))
                idx_exercise = (exercise_value >= continue_value) & (exercise_value > 0)
                payout[idx_exercise] = exercise_value[idx_exercise]
                tau[idx_exercise] = k

            c_fun.update({k: model.c_network})
            g_fun.update({k: model.grad_network})
            print(f"--------------The time step {k} is done!--------------")

        option_value = 1 / num_samples * torch.sum(payout)
        return c_fun, g_fun, option_value





