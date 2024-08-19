import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import ConnectionPatch


def plot_class_geobaskcall(c_fun, cfg, option, plt_size=1000):
    num_time_step = cfg['num_time_step']
    expiration = cfg['expiration']
    dt = expiration / num_time_step
    device = cfg['device']

    _, s_plt = option.sample(plt_size)
    phi_plt = option.phi(s_plt)
    s_plt = s_plt.to(device)
    phi_plt = phi_plt.to(device)

    fig = plt.figure(figsize=(5, 4))
    for k in range(num_time_step - 1, 0, -1):
        with torch.no_grad():
            func = c_fun[k].eval()
            ss_plt_k = torch.cat([phi_plt[:, k].reshape((plt_size, 1)), s_plt[:, :, k]], dim=1)
            continue_plt = func(ss_plt_k).reshape((plt_size,))
            exercise_plt = torch.maximum(phi_plt[:, k], torch.tensor([0.0]))
            idx_plt = (exercise_plt >= continue_plt) & (exercise_plt > 0)

        s_geo_mean = s_plt[:, :, k].log().mean(dim=1).exp()
        plt.scatter(k * dt * torch.ones_like(torch.nonzero(~idx_plt)), s_geo_mean[~idx_plt], c='red', s=2)
        plt.scatter(k * dt * torch.ones_like(torch.nonzero(idx_plt)), s_geo_mean[idx_plt], c='blue', s=2)

    s_star_all = np.loadtxt('./data/ref_boundary.csv')

    time_stamp = np.arange(2/50, 2, 2/50)
    dim = cfg['d']
    if dim == 3:
        plt.scatter(time_stamp, s_star_all[:, 0], c='black', s=10, label='exact boundary')
    elif dim == 20:
        plt.scatter(time_stamp, s_star_all[:, 1], c='black', s=10, label='exact boundary')
    elif dim == 100:
        plt.scatter(time_stamp, s_star_all[:, 2], c='black', s=10, label='exact boundary')
    elif dim == 200:
        plt.scatter(time_stamp, s_star_all[:, 3], c='black', s=10, label='exact boundary')
    elif dim == 500:
        plt.scatter(time_stamp, s_star_all[:, 4], c='black', s=10, label='exact boundary')

    plt.xlabel('time')
    plt.ylabel('geometric average price')
    plt.legend(['continue', 'stop'], loc='upper left')
    plt.show()
    # fig.savefig(f"./save_plot/geobaskcall_{dim}.eps", dpi=300, format='eps')


def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.05, y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)


def plot_value(cfg, option, k, no_trial=1, plt_size=4000):
    # load the model
    c_fun = torch.load(f"./trained_models_geobaskcall/c_models_d{cfg['d']}_N{cfg['num_time_step']}_{no_trial}.pth")
    print("finish loading!")

    num_time_step = cfg['num_time_step']
    expiration = cfg['expiration']
    dt = expiration / num_time_step

    _, s_plt = option.sample(plt_size)
    phi_plt = option.phi(s_plt)

    with torch.no_grad():
        func = c_fun[k].eval()
        ss_plt_k = torch.cat([phi_plt[:, k].reshape((plt_size, 1)), s_plt[:, :, k]], dim=1)
        continue_plt = func(ss_plt_k).reshape((plt_size,))

    # main plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    s_geo_mean = s_plt[:, :, k].log().mean(dim=1).exp()
    ax.scatter(s_geo_mean, continue_plt, c='red', s=1)
    x = torch.arange(70, 200, 0.5)
    y = np.exp(-cfg['r']*k*dt) * torch.maximum(x - cfg['strike'], torch.tensor([0.0]))
    ax.plot(x, y, c='blue')
    plt.xlabel('geometric average price')
    plt.legend([r'$\mathcal{C}_{k,\theta}(X_k)$', r'$g(t_k, X_k)$'], loc='lower right')

    # small plot
    axins = ax.inset_axes((0.1, 0.55, 0.4, 0.4))
    axins.scatter(s_geo_mean, continue_plt, c='red', s=1)
    axins.plot(x, y, c='blue')

    # connect small plot with main plot
    zone_and_linked(ax, axins, 100, 150, x, [y], 'top')

    plt.show()
    # fig.savefig(f"./save_plot/bias_geobaskcall_{cfg['d']}.eps", dpi=300, format='eps')


def plot_grad(cfg, option, k, no_trial=1, plt_size=500):
    # load the model
    c_fun = torch.load(f"./trained_models_geobaskcall/c_models_d{cfg['d']}_N{cfg['num_time_step']}_{no_trial}.pth")
    g_fun = torch.load(f"./trained_models_geobaskcall/g_models_d{cfg['d']}_N{cfg['num_time_step']}_{no_trial}.pth")
    print("finish loading!")

    num_time_step = cfg['num_time_step']
    expiration = cfg['expiration']
    dt = expiration / num_time_step

    # generate samples for plot
    _, s_plt = option.sample(plt_size)
    s_plt_k = s_plt[:, :, k]
    phi_plt = option.phi(s_plt)
    ss_plt_k = torch.cat([phi_plt[:, k].reshape((plt_size, 1)), s_plt_k], dim=1)

    with torch.no_grad():
        func = g_fun[k].eval()
        grad_plt = func(s_plt_k)
        c_func = c_fun[k].eval()
        continue_plt_k = c_func(ss_plt_k).reshape((plt_size,))
    exercise_plt_k = torch.maximum(phi_plt[:, k], torch.tensor([0.0]))
    idx_plt_k = (exercise_plt_k >= continue_plt_k) & (exercise_plt_k > 0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))


    # computed projected delta
    s_geo_mean = s_plt_k.log().mean(dim=1).exp()
    yy = torch.div(torch.sum(torch.mul(grad_plt, s_plt_k), dim=1), s_geo_mean)
    # ax.scatter(s_geo_mean, yy, c='red', s=1)
    ax.scatter(s_geo_mean[~idx_plt_k], yy[~idx_plt_k], c='red', marker='o', s=5)
    ax.scatter(s_geo_mean[idx_plt_k], np.exp(-cfg['r']*k*dt) * torch.ones_like(torch.nonzero(idx_plt_k)),
                c='blue', marker='o', s=5)

    # exact delta
    if k == 25:
        if (cfg['d'] == 20) or (cfg['d'] == 100):
            ss_continue = np.loadtxt(f"./data/ref_d{cfg['d']}_ss_continue.csv")
            delta_continue = np.loadtxt(f"./data/ref_d{cfg['d']}_delta_continue.csv")
            ss_stop = np.loadtxt(f"./data/ref_d{cfg['d']}_ss_stop.csv")
            ax.plot(ss_continue, delta_continue, c='k')
            ax.plot(ss_stop[350:], np.ones_like(ss_stop[350:]), c='k', linewidth=1)

    plt.xlabel('geometric average price')
    plt.legend([r'$\frac{\partial V_k}{\partial\overline{X}_k}= \frac{\partial c(t_k, X_k)}{\partial \overline{X}_k}$',\
                r'$\frac{\partial V_k}{\partial\overline{X}_k}= \frac{\partial g(t_k, X_k)}{\partial \overline{X}_k}$', 'exact delta'],
               loc='upper left', fontsize='large')
    plt.show()
    fig.savefig(f"./save_plot/delta_geobaskcall_{cfg['d']}.eps", dpi=300, format='eps')


def plot_max_call(c_fun, cfg, option, step_plot, plt_size=1000):
    num_time_step = cfg['num_time_step']
    device = cfg['device']

    _, s_plt = option.sample(plt_size)
    phi_plt = option.phi(s_plt)
    s_plt = s_plt.to(device)
    phi_plt = phi_plt.to(device)

    plt.figure()
    for k in range(num_time_step - 1, 0, -1):
        with torch.no_grad():
            func = c_fun[k].eval()
            ss_plt_k = torch.cat([phi_plt[:, k].reshape((plt_size, 1)), s_plt[:, :, k]], dim=1)
            continue_plt = func(ss_plt_k).reshape((plt_size,))
            exercise_plt = torch.maximum(phi_plt[:, k], torch.tensor([0.0]))
            idx_plt = (exercise_plt >= continue_plt) & (exercise_plt > 0)

        if k == step_plot:
            plt.scatter(s_plt[~idx_plt, 0, k], s_plt[~idx_plt, 1, k], c='red', s=2)
            plt.scatter(s_plt[idx_plt, 0, k], s_plt[idx_plt, 1, k], c='blue', s=2)

    plt.show()
