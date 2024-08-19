import argparse
import torch
from sde import get_sde
from train import DeepBSDEOSGeoBaskCall
from plot import plot_class_geobaskcall, plot_value, plot_grad
from bounds import true_lower, true_upper_bs


def get_args():
    parser = argparse.ArgumentParser(description="parameters")
    parser.add_argument('--num_time_step', default=50, type=int, help="number of time steps")
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('--batch_size', default=8192, type=int)
    parser.add_argument('--lr', default=1, type=float, help="base learning rate")
    parser.add_argument('--num_iterations', default=300, type=int, help="number of iterations in each epoch")
    parser.add_argument('--num_epochs', default=1, type=int, help="epoch for training")
    parser.add_argument('--valid_size', default=2 ** 21, type=int, help="number of samples for lower bound")
    parser.add_argument('--upper_size', default=2 ** 15, type=int, help="number of samples for upper bound")
    parser.add_argument('--logging_frequency', default=20, type=int, help="frequency of displaying results")
    parser.add_argument('--device', default='cpu', type=str, help="cpu or cuda")
    # parser.add_argument('--seed', default=1234, type=int, help="random seed")
    # type of option
    parser.add_argument('--option_name', default='GeoBaskCall', type=str, help="types of option")
    parser.add_argument('--strike', default=100, type=float)
    parser.add_argument('--r', default=0, type=float, help="interest rate")
    parser.add_argument('--dividend', default=0.02, type=float, help="dividend yield")
    parser.add_argument('--expiration', default=2, type=float, help="time horizon")
    parser.add_argument('--d', default=20, type=int, help="dimension of stochastic variables")
    parser.add_argument('--s_init', default=100, type=float, help="initial value")
    parser.add_argument('--vol', default=0.25, type=float, help="one volatility for all")
    parser.add_argument('--rho', default=0.75, type=float, help="correlation between the i-th and j-th Brownian motion")
    args = parser.parse_args()
    args = {**vars(args)}
    print(''.join(['='] * 80))
    tplt = "{:^20}\t{:^20}\t{:^20}"
    print(tplt.format("Name", "Value", "Type"))
    for k, v in args.items():
        print(tplt.format(k, v, str(type(v))))
    print(''.join(['='] * 80))
    return args


def train(no_trial=1):
    cfg = get_args()
    option = get_sde(cfg)
    pricer = DeepBSDEOSGeoBaskCall(cfg, option)
    c_fun, g_fun, V0 = pricer.train()
    print(V0)

    # save the trained models
    torch.save(c_fun, f"./trained_models_geobaskcall/c_models_d{cfg['d']}_N{cfg['num_time_step']}_{no_trial}.pth")
    torch.save(g_fun, f"./trained_models_geobaskcall/g_models_d{cfg['d']}_N{cfg['num_time_step']}_{no_trial}.pth")

    with torch.no_grad():
        g_fun[1].eval()
        x0 = 100 * torch.ones((1, cfg['d']))
        x0 = x0.to(torch.device(cfg['device']))
        grad0 = g_fun[1](x0)
        print(grad0)


def test(no_trial=1):
    cfg = get_args()
    option = get_sde(cfg)

    # load the model
    c_fun = torch.load(f"./trained_models_geobaskcall/c_models_d{cfg['d']}_N{cfg['num_time_step']}_{no_trial}.pth")
    g_fun = torch.load(f"./trained_models_geobaskcall/g_models_d{cfg['d']}_N{cfg['num_time_step']}_{no_trial}.pth")
    print("finish loading!")

    plot_class_geobaskcall(c_fun, cfg, option)

    # compute lower bound
    option_price, h_lower = true_lower(cfg, c_fun, cfg['valid_size'])
    print(f"lower bound is {option_price:.4f}, confidence interval:{h_lower:.4f}")

    # compute upper bound
    c1_fun = c_fun[1]
    option_upper, h_upper = true_upper_bs(cfg, g_fun, c1_fun, scale=32, upper_size=cfg['upper_size'])
    print(f"upper bound is {option_upper:.4f}, confidence interval:{h_upper:.4f}")

    print(f"95% confidence interval is [{option_price - h_lower:.4f}, {option_upper + h_upper:.4f}].")
    print(f"relative difference percentage is {(option_upper - option_price + h_lower + h_upper) / option_price:.4f}")


if __name__ == '__main__':
    num = 1
    # train(no_trial=num)
    # test(no_trial=num)

    cfg = get_args()
    option = get_sde(cfg)
    # plot_value(cfg, option, k=49, no_trial=num, plt_size=1000)
    plot_grad(cfg, option, k=25, no_trial=num, plt_size=500)









