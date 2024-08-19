import torch
import numpy as np


class SDE(object):
    def __init__(self, dim, expiration, num_time_step):
        self.dim = dim
        self.time_horizon = expiration
        self.num_time_step = num_time_step
        self.dt = self.time_horizon / self.num_time_step
        self.sqrt_dt = torch.sqrt(torch.tensor(self.dt))


class GeoBaskCall(SDE):
    def __init__(self, dim, expiration, num_time_step, cfg):
        super(GeoBaskCall, self).__init__(dim, expiration, num_time_step)
        self.s_init = torch.ones(self.dim) * cfg['s_init']
        self.rate = cfg['r']
        self.dividend = cfg['dividend']
        self.volatility = cfg['vol']
        self.correlation = torch.eye(self.dim) * (1 - cfg['rho']) + torch.ones((self.dim, self.dim)) * cfg['rho']
        self.strike = cfg['strike']

    def sample(self, num_sample):
        eigenvalues, eigenvectors = torch.linalg.eigh(self.correlation)
        eigenvalues = torch.diag(eigenvalues)
        c = eigenvectors @ torch.sqrt(eigenvalues)
        z = torch.randn([num_sample, self.dim, self.num_time_step], dtype=torch.float32)
        dw_sample = torch.matmul(c, z) * self.sqrt_dt                  # correlated Brownian motion
        s_sample = torch.zeros([num_sample, self.dim, self.num_time_step + 1], dtype=torch.float32)
        s_sample[:, :, 0] = torch.ones([num_sample, self.dim]) * self.s_init

        log_price_drift = self.rate - self.dividend - .5 * self.volatility ** 2

        log_price = torch.zeros([num_sample, self.dim], dtype=torch.float32)
        for i in range(self.num_time_step):
            log_price += log_price_drift * self.dt + self.volatility * dw_sample[:, :, i]
            s_sample[:, :, i + 1] = torch.exp(log_price) * self.s_init
        return dw_sample, s_sample

    def payoff(self, s):
        ans = torch.maximum(s.log().mean(dim=1).exp() - self.strike, torch.tensor([0.0]))
        discount = torch.ones((s.shape[0], s.shape[2]))
        discount[:, 0] = torch.zeros((s.shape[0]))
        discount = torch.exp(-self.rate * self.dt * torch.cumsum(discount, dim=1))
        return discount * ans

    def phi(self, s):
        """
        :return: phi(s), where payoff = max(phi(s), 0)
        """
        ans = s.log().mean(dim=1).exp() - self.strike
        discount = torch.ones((s.shape[0], s.shape[2]))
        discount[:, 0] = torch.zeros((s.shape[0]))
        discount = torch.exp(-self.rate * self.dt * torch.cumsum(discount, dim=1))
        return discount * ans

    def phi_partial(self, s, delta_t):
        """
        compute phi(s) using paths on partial time points with time difference delta_t.
        :return: phi(s), where payoff = max(phi(s), 0)
        """
        ans = s.log().mean(dim=1).exp() - self.strike
        discount = torch.ones((s.shape[0], s.shape[2]))
        discount[:, 0] = torch.zeros((s.shape[0]))
        discount = torch.exp(-self.rate * delta_t * torch.cumsum(discount, dim=1))
        return discount * ans


class Strangle(SDE):
    def __init__(self, dim, expiration, num_time_step, cfg):
        super(Strangle, self).__init__(dim, expiration, num_time_step)
        self.s_init = torch.ones(self.dim) * cfg['s_init']
        self.rate = cfg['r']
        self.dividend = cfg['dividend']
        self.volatility = torch.tensor([[0.3024,   0.1354,   0.0722,   0.1367,   0.1641],
                                        [0.1354,   0.2270,   0.0613,   0.1264,   0.1610],
                                        [0.0722,   0.0613,   0.0717,   0.0884,   0.0699],
                                        [0.1367,   0.1264,   0.0884,   0.2937,   0.1394],
                                        [0.1641,   0.1610,   0.0699,   0.1394,   0.2535]])

    def sample(self, num_sample):
        dw_sample = self.sqrt_dt * torch.randn([num_sample, self.dim, self.num_time_step], dtype=torch.float32)
        s_sample = torch.zeros([num_sample, self.dim, self.num_time_step + 1], dtype=torch.float32)
        s_sample[:, :, 0] = torch.ones([num_sample, self.dim]) * self.s_init

        log_price_drift = self.rate - self.dividend - .5 * torch.diag(self.volatility @ self.volatility.T)

        log_price = torch.zeros([num_sample, self.dim], dtype=torch.float32)
        for i in range(self.num_time_step):
            log_price += log_price_drift * self.dt + dw_sample[:, :, i] @ self.volatility.T
            s_sample[:, :, i + 1] = torch.exp(log_price) * self.s_init
        return dw_sample, s_sample

    def payoff(self, s):
        ans = torch.maximum(15 - torch.maximum(25 - abs(s.mean(dim=1) - 100), torch.tensor([0.0])), torch.tensor([0.0]))
        discount = torch.ones((s.shape[0], s.shape[2]))
        discount[:, 0] = torch.zeros((s.shape[0]))
        discount = torch.exp(-self.rate * self.dt * torch.cumsum(discount, dim=1))
        return discount * ans

    def phi(self, s):
        """
        :return: phi(s), where payoff = max(phi(s), 0)
        """
        ans = 15 - torch.maximum(25 - abs(s.mean(dim=1) - 100), torch.tensor([0.0]))
        discount = torch.ones((s.shape[0], s.shape[2]))
        discount[:, 0] = torch.zeros((s.shape[0]))
        discount = torch.exp(-self.rate * self.dt * torch.cumsum(discount, dim=1))
        return discount * ans

    def phi_partial(self, s, delta_t):
        """
        compute phi(s) using paths on partial time points with time difference delta_t.
        :return: phi(s), where payoff = max(phi(s), 0)
        """
        ans = 15 - torch.maximum(25 - abs(s.mean(dim=1) - 100), torch.tensor([0.0]))
        discount = torch.ones((s.shape[0], s.shape[2]))
        discount[:, 0] = torch.zeros((s.shape[0]))
        discount = torch.exp(-self.rate * delta_t * torch.cumsum(discount, dim=1))
        return discount * ans


class Heston(SDE):
    def __init__(self, dim, expiration, num_time_step, cfg):
        super(Heston, self).__init__(dim, expiration, num_time_step)
        self.s_init = cfg['s_init']
        self.v_init = cfg['v_init']
        self.rate = cfg['r']
        self.strike = cfg['strike']
        self.cor = cfg['correlation']
        self.sqrt_cor = torch.sqrt(torch.tensor(1 - self.cor ** 2))
        self.kappa = cfg['kappa']
        self.theta = cfg['theta']
        self.nu = cfg['nu']

    def sample(self, num_sample):
        dw_sample = self.sqrt_dt * torch.randn([num_sample, self.dim, self.num_time_step], dtype=torch.float32)
        xv_sample = torch.zeros([num_sample, self.dim, self.num_time_step + 1], dtype=torch.float32)
        xv_sample[:, 1, 0] = torch.ones(num_sample) * self.v_init

        for i in range(self.num_time_step):
            xv_sample[:, 0, i+1] = xv_sample[:, 0, i] + (self.rate - .5 * xv_sample[:, 1, i]) * self.dt \
                + torch.sqrt(xv_sample[:, 1, i]) * (self.cor * dw_sample[:, 0, i] + self.sqrt_cor * dw_sample[:, 1, i])
            xv_sample[:, 1, i+1] = xv_sample[:, 1, i] + self.kappa * (self.theta - xv_sample[:, 1, i]) * self.dt \
                + self.nu * torch.sqrt(xv_sample[:, 1, i]) * dw_sample[:, 0, i]
            xv_sample[:, 1, i+1] = xv_sample[:, 1, i+1].abs()
        return dw_sample, xv_sample

    def payoff(self, xv):
        ans = torch.maximum(self.strike - xv[:, 0, :].exp() * self.s_init, torch.tensor([0.0]))
        discount = torch.ones((xv.shape[0], xv.shape[2]))
        discount[:, 0] = torch.zeros((xv.shape[0]))
        discount = torch.exp(-self.rate * self.dt * torch.cumsum(discount, dim=1))
        return discount * ans

    def phi(self, xv):
        """
        :return: phi(s), where payoff = max(phi(s), 0)
        """
        ans = self.strike - xv[:, 0, :].exp() * self.s_init
        discount = torch.ones((xv.shape[0], xv.shape[2]))
        discount[:, 0] = torch.zeros((xv.shape[0]))
        discount = torch.exp(-self.rate * self.dt * torch.cumsum(discount, dim=1))
        return discount * ans

    def phi_partial(self, xv, delta_t):
        """
        compute phi(s) using paths on partial time points with time difference delta_t.
        :return: phi(s), where payoff = max(phi(s), 0)
        """
        ans = self.strike - xv[:, 0, :].exp() * self.s_init
        discount = torch.ones((xv.shape[0], xv.shape[2]))
        discount[:, 0] = torch.zeros((xv.shape[0]))
        discount = torch.exp(-self.rate * delta_t * torch.cumsum(discount, dim=1))
        return discount * ans


class MaxCall(SDE):
    def __init__(self, dim, expiration, num_time_step, cfg):
        super(MaxCall, self).__init__(dim, expiration, num_time_step)
        self.s_init = torch.ones(self.dim) * cfg['s_init']
        self.rate = cfg['r']
        self.dividend = cfg['dividend']
        self.volatility = cfg['vol']
        self.correlation = torch.eye(self.dim)
        self.strike = cfg['strike']

    def sample(self, num_sample):
        dw_sample = self.sqrt_dt * torch.randn([num_sample, self.dim, self.num_time_step], dtype=torch.float32)
        s_sample = torch.zeros([num_sample, self.dim, self.num_time_step + 1], dtype=torch.float32)
        s_sample[:, :, 0] = torch.ones([num_sample, self.dim]) * self.s_init

        log_price_drift = self.rate - self.dividend - .5 * self.volatility ** 2

        log_price = torch.zeros([num_sample, self.dim], dtype=torch.float32)
        for i in range(self.num_time_step):
            log_price += log_price_drift * self.dt + self.volatility * dw_sample[:, :, i]
            s_sample[:, :, i + 1] = torch.exp(log_price) * self.s_init
        return dw_sample, s_sample

    def payoff(self, s):
        ans = torch.maximum(torch.max(s - self.strike, dim=1)[0], torch.tensor([0.0]))
        discount = torch.ones((s.shape[0], s.shape[2]))
        discount[:, 0] = torch.zeros((s.shape[0]))
        discount = torch.exp(-self.rate * self.dt * torch.cumsum(discount, dim=1))
        return discount * ans

    def phi(self, s):
        """
        :return: phi(s), where payoff = max(phi(s), 0)
        """
        ans = torch.max(s - self.strike, dim=1)[0]
        discount = torch.ones((s.shape[0], s.shape[2]))
        discount[:, 0] = torch.zeros((s.shape[0]))
        discount = torch.exp(-self.rate * self.dt * torch.cumsum(discount, dim=1))
        return discount * ans

    def phi_partial(self, s, delta_t):
        """
        compute phi(s) using paths on partial time points with time difference delta_t.
        :return: phi(s), where payoff = max(phi(s), 0)
        """
        ans = torch.max(s - self.strike, dim=1)[0]
        discount = torch.ones((s.shape[0], s.shape[2]))
        discount[:, 0] = torch.zeros((s.shape[0]))
        discount = torch.exp(-self.rate * delta_t * torch.cumsum(discount, dim=1))
        return discount * ans


def all_seed(seed=1):
    """
    set up random seed for all
    """
    np.random.seed(seed)
    torch.manual_seed(seed)        # config for CPU
    torch.cuda.manual_seed(seed)   # config for GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_sde(cfg):
    try:
        # if cfg['seed'] != 0:
        #     all_seed(seed=cfg['seed'])
        return globals()[cfg['option_name']](cfg['d'], cfg['expiration'], cfg['num_time_step'], cfg)
    except KeyError:
        raise KeyError("Option type required not found. Please try others.")

