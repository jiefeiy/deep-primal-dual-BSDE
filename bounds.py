import time
import torch
import numpy as np
from sde import get_sde
from solver import MLP
from scipy.stats import norm, sem


def true_lower(cfg, c_fun, lower_size=2 ** 20):
    """
    compute true lower bound
    """
    option = get_sde(cfg)
    num_time_step = cfg['num_time_step']

    if cfg['d'] <= 50:
        test_start_time = time.time()
        _, x_valid = option.sample(lower_size)
        phi_valid = option.phi(x_valid)
        x_valid = x_valid.to(cfg['device'])
        phi_valid = phi_valid.to(cfg['device'])

        payout_valid = torch.maximum(phi_valid[:, num_time_step], torch.tensor([0.0]))
        for k in range(num_time_step - 1, 0, -1):
            with torch.no_grad():
                func = c_fun[k].eval()
                xx_valid_k = torch.cat([phi_valid[:, k].reshape((lower_size, 1)), x_valid[:, :, k]], dim=1)
                continue_valid = func(xx_valid_k).reshape((lower_size,))
                exercise_valid = torch.maximum(phi_valid[:, k], torch.tensor([0.0]))
                idx_valid = (exercise_valid >= continue_valid) & (exercise_valid > 0)
                payout_valid[idx_valid] = exercise_valid[idx_valid]
        price_lower_bound, se = torch.mean(payout_valid), sem(payout_valid)

        confidence = 0.95
        z = norm.ppf((1 + confidence) / 2)
        print(f"-------lower bound evaluation time is {time.time() - test_start_time:.2f}")
    else:
        test_start_time = time.time()
        payout_valid_all = torch.tensor([])
        valid_size = round(lower_size / 4)    # memory efficiency
        for i in range(4):
            _, x_valid = option.sample(valid_size)
            phi_valid = option.phi(x_valid)
            x_valid = x_valid.to(cfg['device'])
            phi_valid = phi_valid.to(cfg['device'])

            payout_valid = torch.maximum(phi_valid[:, num_time_step], torch.tensor([0.0]))
            for k in range(num_time_step - 1, 0, -1):
                with torch.no_grad():
                    func = c_fun[k].eval()
                    xx_valid_k = torch.cat([phi_valid[:, k].reshape((valid_size, 1)), x_valid[:, :, k]], dim=1)
                    continue_valid = func(xx_valid_k).reshape((valid_size,))
                    exercise_valid = torch.maximum(phi_valid[:, k], torch.tensor([0.0]))
                    idx_valid = (exercise_valid >= continue_valid) & (exercise_valid > 0)
                    payout_valid[idx_valid] = exercise_valid[idx_valid]
            payout_valid_all = torch.cat([payout_valid_all, payout_valid])

        price_lower_bound, se = torch.mean(payout_valid_all), sem(payout_valid_all)
        confidence = 0.95
        z = norm.ppf((1 + confidence) / 2)
        print(f"-------lower bound evaluation time is {time.time() - test_start_time:.2f}")
    return price_lower_bound, se * z


def true_upper_bs(cfg, g_fun, c1_fun, scale=32, upper_size=2 ** 15):
    """
    true upper bound via non-nested Monte Carlo
    """
    delta_t = cfg['expiration'] / cfg['num_time_step']                              # original \Delta t
    num_fine_grid = scale * cfg['num_time_step']
    cfg['num_time_step'] = num_fine_grid
    vol = cfg['vol']
    option_upper = get_sde(cfg)

    if cfg['d'] <= 50:
        upper_start_time = time.time()
        dw_upper, s_upper = option_upper.sample(upper_size)                             # generate samples on fine time grid
        # only need payoff on coarse grid
        phi_required = option_upper.phi_partial(s_upper[:, :, 0:(num_fine_grid + 1):scale], delta_t)
        dm_mat = torch.zeros(size=(upper_size, num_fine_grid), dtype=torch.float32, device=cfg['device'])

        for j in range(scale, num_fine_grid):
            k = np.floor(j / scale)
            z = vol * torch.mul(s_upper[:, :, j], dw_upper[:, :, j])  # \sigma(S_k) dW_k
            with torch.no_grad():
                grad_model = g_fun[k].eval()
                g = grad_model(s_upper[:, :, j])
            dm_mat[:, j] = torch.sum(torch.mul(g, z), dim=1)

        with torch.no_grad():
            c1_model = c1_fun.eval()
            ss_valid_1 = torch.cat([phi_required[:, 1].reshape(upper_size, 1), s_upper[:, :, scale]], dim=1)
            v1 = torch.maximum(c1_model(ss_valid_1).reshape((upper_size,)), phi_required[:, 1])  # should be positive
            dm_mat[:, scale - 1] = v1 - torch.mean(v1)
        m_total = torch.cumsum(dm_mat, dim=1)
        m_total = torch.cat([torch.zeros(size=(upper_size, 1), dtype=torch.float32).to(cfg['device']), m_total], dim=1)
        martingale = m_total[:, 0:(num_fine_grid + 1):scale]

        v_mat = torch.maximum(phi_required, torch.tensor([0.0])) - martingale
        m_star, k_star = torch.max(v_mat, dim=1)
        upper_data = v_mat[range(upper_size), k_star]
        price_upper_bound, se = torch.mean(upper_data), sem(upper_data)

        confidence = 0.95
        z = norm.ppf((1 + confidence) / 2)
        print(f"-------upper bound evaluation time is {time.time() - upper_start_time:.2f}")
    else:
        upper_start_time = time.time()
        upper_data_all = torch.tensor([])
        valid_size = round(upper_size / 2)   # memory efficiency
        for i in range(2):
            dw_upper, s_upper = option_upper.sample(valid_size)  # generate samples on fine time grid
            # only need payoff on coarse grid
            phi_required = option_upper.phi_partial(s_upper[:, :, 0:(num_fine_grid + 1):scale], delta_t)
            dm_mat = torch.zeros(size=(valid_size, num_fine_grid), dtype=torch.float32, device=cfg['device'])

            for j in range(scale, num_fine_grid):
                k = np.floor(j / scale)
                z = vol * torch.mul(s_upper[:, :, j], dw_upper[:, :, j])  # \sigma(S_k) dW_k
                with torch.no_grad():
                    grad_model = g_fun[k].eval()
                    g = grad_model(s_upper[:, :, j])
                dm_mat[:, j] = torch.sum(torch.mul(g, z), dim=1)

            with torch.no_grad():
                c1_model = c1_fun.eval()
                ss_valid_1 = torch.cat([phi_required[:, 1].reshape(valid_size, 1), s_upper[:, :, scale]], dim=1)
                v1 = torch.maximum(c1_model(ss_valid_1).reshape((valid_size,)), phi_required[:, 1])  # should be positive
                dm_mat[:, scale - 1] = v1 - torch.mean(v1)
            m_total = torch.cumsum(dm_mat, dim=1)
            m_total = torch.cat([torch.zeros(size=(valid_size, 1), dtype=torch.float32).to(cfg['device']), m_total], dim=1)
            martingale = m_total[:, 0:(num_fine_grid + 1):scale]

            v_mat = torch.maximum(phi_required, torch.tensor([0.0])) - martingale
            m_star, k_star = torch.max(v_mat, dim=1)
            upper_data = v_mat[range(valid_size), k_star]
            upper_data_all = torch.cat([upper_data_all, upper_data])
        price_upper_bound, se = torch.mean(upper_data_all), sem(upper_data_all)

        confidence = 0.95
        z = norm.ppf((1 + confidence) / 2)
        print(f"-------upper bound evaluation time is {time.time() - upper_start_time:.2f}")
    return price_upper_bound, se * z


def true_upper_strangle(cfg, g_fun, c1_fun, scale=32, upper_size=2 ** 15):
    """
    true upper bound via non-nested Monte Carlo
    """
    delta_t = cfg['expiration'] / cfg['num_time_step']                              # original \Delta t
    num_fine_grid = scale * cfg['num_time_step']
    cfg['num_time_step'] = num_fine_grid
    vol = torch.tensor([[0.3024,   0.1354,   0.0722,   0.1367,   0.1641],
                        [0.1354,   0.2270,   0.0613,   0.1264,   0.1610],
                        [0.0722,   0.0613,   0.0717,   0.0884,   0.0699],
                        [0.1367,   0.1264,   0.0884,   0.2937,   0.1394],
                        [0.1641,   0.1610,   0.0699,   0.1394,   0.2535]])
    option_upper = get_sde(cfg)

    upper_start_time = time.time()
    dw_upper, s_upper = option_upper.sample(upper_size)                             # generate samples on fine time grid
    # only need payoff on coarse grid
    phi_required = option_upper.phi_partial(s_upper[:, :, 0:(num_fine_grid + 1):scale], delta_t)
    dm_mat = torch.zeros(size=(upper_size, num_fine_grid), dtype=torch.float32, device=cfg['device'])

    for j in range(scale, num_fine_grid):
        k = np.floor(j / scale)
        z = torch.mul(s_upper[:, :, j], dw_upper[:, :, j] @ vol.T)  # \sigma(S_k) dW_k
        with torch.no_grad():
            grad_model = g_fun[k].eval()
            g = grad_model(s_upper[:, :, j])
        dm_mat[:, j] = torch.sum(torch.mul(g, z), dim=1)

    with torch.no_grad():
        c1_model = c1_fun.eval()
        ss_valid_1 = torch.cat([phi_required[:, 1].reshape(upper_size, 1), s_upper[:, :, scale]], dim=1)
        v1 = torch.maximum(c1_model(ss_valid_1).reshape((upper_size,)), phi_required[:, 1])  # should be positive
        dm_mat[:, scale - 1] = v1 - torch.mean(v1)
    m_total = torch.cumsum(dm_mat, dim=1)
    m_total = torch.cat([torch.zeros(size=(upper_size, 1), dtype=torch.float32).to(cfg['device']), m_total], dim=1)
    martingale = m_total[:, 0:(num_fine_grid + 1):scale]

    v_mat = torch.maximum(phi_required, torch.tensor([0.0])) - martingale
    m_star, k_star = torch.max(v_mat, dim=1)
    upper_data = v_mat[range(upper_size), k_star]
    price_upper_bound, se = torch.mean(upper_data), sem(upper_data)

    confidence = 0.95
    z = norm.ppf((1 + confidence) / 2)
    print(f"-------upper bound evaluation time is {time.time() - upper_start_time:.2f}")
    return price_upper_bound, se * z


def true_upper_heston(cfg, g_fun, c1_fun, scale=32, upper_size=2 ** 15):
    """
    true upper bound via non-nested Monte Carlo
    """
    delta_t = cfg['expiration'] / cfg['num_time_step']                              # original \Delta t
    num_fine_grid = scale * cfg['num_time_step']
    cfg['num_time_step'] = num_fine_grid
    cor = cfg['correlation']
    sqrt_cor = torch.sqrt(torch.tensor(1 - cor ** 2))
    nu = cfg['nu']
    vol_mat_T = torch.tensor([[cor, nu], [sqrt_cor, 0.0]])
    option_upper = get_sde(cfg)

    upper_start_time = time.time()
    dw_upper, xv_upper = option_upper.sample(upper_size)                        # generate samples on fine time grid
    # only need payoff on coarse grid
    phi_required = option_upper.phi_partial(xv_upper[:, :, 0:(num_fine_grid + 1):scale], delta_t)
    dm_mat = torch.zeros(size=(upper_size, num_fine_grid), dtype=torch.float32, device=cfg['device'])

    for j in range(scale, num_fine_grid):
        k = np.floor(j / scale)
        z = torch.sqrt(xv_upper[:, 1, j]).reshape((upper_size, 1)) * (dw_upper[:, :, j] @ vol_mat_T)  # \sigma(S_k) dW_k
        with torch.no_grad():
            grad_model = g_fun[k].eval()
            g = grad_model(xv_upper[:, :, j])
        dm_mat[:, j] = torch.sum(torch.mul(g, z), dim=1)

    with torch.no_grad():
        c1_model = c1_fun.eval()
        xx_valid_1 = torch.cat([phi_required[:, 1].reshape(upper_size, 1), xv_upper[:, :, scale]], dim=1)
        v1 = torch.maximum(c1_model(xx_valid_1).reshape((upper_size,)), phi_required[:, 1])  # should be positive
        dm_mat[:, scale - 1] = v1 - torch.mean(v1)
    m_total = torch.cumsum(dm_mat, dim=1)
    m_total = torch.cat([torch.zeros(size=(upper_size, 1), dtype=torch.float32).to(cfg['device']), m_total], dim=1)
    martingale = m_total[:, 0:(num_fine_grid + 1):scale]

    v_mat = torch.maximum(phi_required, torch.tensor([0.0])) - martingale
    m_star, k_star = torch.max(v_mat, dim=1)
    upper_data = v_mat[range(upper_size), k_star]
    price_upper_bound, se = torch.mean(upper_data), sem(upper_data)

    confidence = 0.95
    z = norm.ppf((1 + confidence) / 2)
    print(f"-------upper bound evaluation time is {time.time() - upper_start_time:.2f}")
    return price_upper_bound, se * z


