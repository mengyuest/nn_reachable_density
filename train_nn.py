import os, sys
import time
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
from torch.utils.tensorboard import SummaryWriter
import sub_utils
from multiprocessing.pool import Pool

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.relu = nn.ReLU()
        self.linear_list = nn.ModuleList()
        self.linear_list.append(nn.Linear(args.input_dim, args.hiddens[0]))
        for i, hidden in enumerate(args.hiddens):
            in_dim = args.hiddens[i]
            if i == len(args.hiddens) - 1:
                if args.only_density_arch:
                    out_dim = 1
                elif args.only_dynamics_arch:
                    out_dim = args.input_dim - 1
                else:
                    out_dim = args.input_dim
            else:
                out_dim = args.hiddens[i + 1]
            self.linear_list.append(nn.Linear(in_dim, out_dim))

        if self.args.normalize:
            self.in_means=torch.from_numpy(self.args.in_means).float()
            self.in_stds=torch.from_numpy(self.args.in_stds).float()
            self.out_means=torch.from_numpy(self.args.out_means).float()
            self.out_stds=torch.from_numpy(self.args.out_stds).float()

    def update_device(self):
        if self.args.normalize:
            # device=torch.device('cuda:0')
            self.in_means=self.in_means.cuda() #.to(device)
            self.in_stds=self.in_stds.cuda() #.to(device)
            self.out_means=self.out_means.cuda() #.to(device)
            self.out_stds=self.out_stds.cuda() #.to(device)

    def forward(self, ori_x):
        if self.args.normalize:
            x = (ori_x - self.in_means) / self.in_stds
        else:
            x = ori_x

        for i, hidden in enumerate(self.args.hiddens):
            x = self.relu(self.linear_list[i](x))
        x = self.linear_list[len(self.args.hiddens)](x)


        if self.args.t_struct:
            if self.args.only_density_arch:
                t_part = ori_x[:, -1:]
                log_rho = x[:, 0:1] * t_part
                x = log_rho
            elif self.args.only_dynamics_arch==False:
                t_part = ori_x[:, -1:]
                log_rho = x[:, 0:1] * t_part
                est_x = x[:, 1:]
                x = torch.cat([log_rho, est_x], dim=-1)

        if self.args.normalize:
            out_x = x * self.out_stds + self.out_means
        else:
            out_x = x

        return out_x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--exp_mode', type=str, choices= \
        ['robot', 'dint', 'toon', 'car', 'quad', 'gcas', 'acc', 'pend', 'vdp', 'circ', "kop"])
    parser.add_argument('--exp_name', type=str, default="exp")
    parser.add_argument('--hiddens', type=int, nargs="+", default=[256, 256, 256])

    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--eval_freq', type=int, default=10000)
    parser.add_argument('--random_seed', type=int, default=1007)

    parser.add_argument('--beta', type=float, default=0.2)  # trade-off between mse loss and pde loss
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--dt', type=float, default=None)  # discrete time for neural network
    parser.add_argument('--max_len', type=int, default=None)
    parser.add_argument('--train_data_path', type=str, default="train_data_vdp_2steps.npy")
    parser.add_argument('--pretrained_path', type=str, default=None)

    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--wrap_log1', action='store_true', default=False)
    parser.add_argument('--wrap_log2', action='store_true', default=False)
    parser.add_argument('--nx', type=int, default=None)
    parser.add_argument('--ny', type=int, default=None)

    parser.add_argument('--eval_mode', action='store_true', default=False)

    parser.add_argument('--x_min', type=float, default=None)
    parser.add_argument('--x_max', type=float, default=None)
    parser.add_argument('--y_min', type=float, default=None)
    parser.add_argument('--y_max', type=float, default=None)

    parser.add_argument('--more_error', action='store_true', default=False)  # TODO

    parser.add_argument('--dyna_weight', type=float ,default=1.0)

    parser.add_argument('--x_index', type=int, default=0)
    parser.add_argument('--y_index', type=int, default=1)
    parser.add_argument('--x_label', type=str, default="x")
    parser.add_argument('--y_label', type=str, default="y")
    parser.add_argument('--n_xticks', type=int, default=5)
    parser.add_argument('--n_yticks', type=int, default=8)

    parser.add_argument('--less_t', action='store_true', default=False)
    parser.add_argument('--train_dyna_only', action='store_true', default=False)
    parser.add_argument('--train_density_only', action='store_true', default=False)

    parser.add_argument('--t_struct', action='store_true', default=False)

    parser.add_argument('--log_density', action='store_true', default=False)
    parser.add_argument('--show_stat', action='store_true', default=False)  # TODO
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--viz_in_unlog', action='store_true', default=False)

    parser.add_argument('--change_after', type=int, default=None)
    parser.add_argument('--new_show_stat', action='store_true', default=False)
    parser.add_argument('--new_dyna_weight', type=float, default=None)
    parser.add_argument('--new_lr', type=float, default=None)
    parser.add_argument('--debug_batchsize', type=int, default=None)
    parser.add_argument('--ratio_loss', action='store_true', default=False)
    parser.add_argument('--l1_loss', action='store_true', default=False)

    parser.add_argument('--init_x_weight', type=float, default=None)
    parser.add_argument('--to_see_dens', action='store_true', default=False)

    parser.add_argument('--scale_with_output', action='store_true', default=False)
    parser.add_argument('--no_dens_std', action='store_true', default=False)

    parser.add_argument('--only_density_arch', action='store_true', default=False)
    parser.add_argument('--only_dynamics_arch', action='store_true', default=False)
    parser.add_argument('--gaussian_init', action='store_true')
    parser.add_argument('--gaussian_mean', nargs="+", type=float, default=[1.0, 0.0, 0.0])
    parser.add_argument('--gaussian_vars', nargs="+", type=float, default=[0.0625, 0.25, 0.25])

    parser.add_argument('--train_rate', type=float, default=None)

    return parser.parse_args()

def get_data_tensor_from_numpy(args, filename):
    data={"train":{}, "test":{}}
    raw_data = np.load(os.path.join("cache", args.train_data_path, filename), allow_pickle=True)['arr_0'].item()
    keys = ['s_list', 'rho_list', 'nabla_list', 't_list']  # (n_trajs, nt, dim)
    if args.max_len is not None:
        for k in keys:
            raw_data[k] = raw_data[k][:, :args.max_len]
            if args.debug_batchsize is not None:
                raw_data[k] = raw_data[k][:args.debug_batchsize]
    if args.log_density:
        raw_data["rho_list"] = np.log(raw_data["rho_list"])

    total_num_traj = raw_data["s_list"].shape[0]
    train_num_traj = int(total_num_traj * 0.8)
    test_num_traj = total_num_traj - train_num_traj

    if args.train_rate is not None:
        train_num_traj = int(args.train_rate * train_num_traj)
        print("Train:%d  Test:%d"%(train_num_traj, test_num_traj))

    if args.normalize:
        in_means, in_stds, out_means, out_stds = sub_utils.get_data_stat(raw_data, train_num_traj)
        if args.scale_with_output:
            in_means[:-1] = out_means[1:]
            in_stds[:-1] = out_stds[1:]
        if args.no_dens_std:
            out_stds[0] = 1.0

        if args.only_dynamics_arch:
            out_stds = out_stds[1:]
            out_means = out_means[1:]
        elif args.only_density_arch:
            out_stds = out_stds[0:1]
            out_means = out_means[0:1]

        print("Normalization")
        print("  \tmeans\tstds\tmeans\tstds")
        for i in range(in_means.shape[0]):
            if args.only_density_arch and i>=1:
                print("%02d\t%9.4f\t%9.4f" % (i, in_means[i], in_stds[i]))
            elif args.only_dynamics_arch:
                if i>=in_means.shape[0]-1:
                    print("%02d\t%9.4f\t%9.4f" % (i, in_means[i], in_stds[i]))
                else:
                    print("%02d\t%9.4f\t%9.4f\t%9.4f\t%9.4f" % (i, in_means[i], in_stds[i], out_means[i], out_stds[i]))
            else:
                print("%02d\t%9.4f\t%9.4f\t%9.4f\t%9.4f"%(i, in_means[i], in_stds[i], out_means[i], out_stds[i]))

    else:
        in_means = in_stds = out_means = out_stds = None


    for k in keys:
        k_less=k.split("_")[0]
        data["train"][k_less] = torch.from_numpy(raw_data[k][:train_num_traj]).float()
        data["test"][k_less] = torch.from_numpy(raw_data[k][-test_num_traj:]).float()
        data["train"][k_less] = torch.swapaxes(data["train"][k_less], 0, 1)
        data["test"][k_less] = torch.swapaxes(data["test"][k_less], 0, 1)

        # TODO print(data["test"][k_less].shape) # (T, N, ndim)

        if k != "s_list":
            data["train"][k_less] = data["train"][k_less].unsqueeze(dim=-1)
            data["test"][k_less] = data["test"][k_less].unsqueeze(dim=-1)

    init_rho={}
    if args.exp_mode == "kop":
        from scipy.stats import multivariate_normal as mm
        init_rho["train"] = mm.pdf(data["train"]["s"][0,:].detach().cpu().numpy(), mean=args.gaussian_mean, cov=args.gaussian_vars)
        init_rho["train"] = torch.from_numpy(init_rho["train"]).float()

        init_rho["test"] = mm.pdf(data["test"]["s"][0,:].detach().cpu().numpy(), mean=args.gaussian_mean, cov=args.gaussian_vars)
        init_rho["test"] = torch.from_numpy(init_rho["test"]).float()
    else:
        init_rho["train"] = None
        init_rho["test"] = None

    return data, data["train"], data["test"], in_means, in_stds, out_means, out_stds, init_rho["train"], init_rho["test"]

def plot_loss_curve(nt, losses, fig_name):
    ts = range(nt)
    plt.plot(ts, torch.mean(losses.reshape(nt, -1), dim=-1).detach().cpu().numpy())
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_heatmap(fig_name, heat, args):
    nx = args.nx
    ny = args.ny
    im = plt.imshow(heat, origin='lower', cmap=cm.inferno)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, format="%.2f")
    ax = plt.gca()
    ax.set_xticks(np.linspace(0, nx, args.n_xticks))
    ax.set_xticklabels(["%.3f" % xx for xx in np.linspace(args.x_min, args.x_max, args.n_xticks)])
    ax.set_yticks(np.linspace(0, ny, args.n_yticks))
    ax.set_yticklabels(["%.3f" % xx for xx in np.linspace(args.y_min, args.y_max, args.n_yticks)])
    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_scatter(fig_name, xs1, ys1, c1, xs2, ys2, c2, args):
    plt.scatter(xs1, ys1, color=c1, alpha=0.5, s=1.0)
    plt.scatter(xs2, ys2, color=c2, alpha=0.5, s=1.0)
    # plt.axis('scaled')
    ax = plt.gca()
    ax.set_aspect(1.0 / ((args.y_max-args.y_min)/(args.x_max-args.x_min)), adjustable='box')
    plt.xlim(args.x_min, args.x_max)
    plt.ylim(args.y_min, args.y_max)
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    plt.close()

def gridify(xs, xmin, xmax, nx):
    return torch.clamp(((xs - xmin) / ((xmax-xmin)/nx)).int(), 0, nx-1).numpy()

def plot_t_density_particles(input_data):
    epi, t, test_gain_dyna_full_plain, test_gt_x_plain, test_gtr_plain, n_test, input_dim, test_ri, args = input_data

    nx = args.nx
    ny = args.ny
    if args.viz_in_unlog:
        viz_gt = np.zeros((ny, nx))
        viz_est = np.zeros((ny, nx))
    else:
        viz_gt = np.ones((ny, nx)) * -10
        viz_est = np.ones((ny, nx)) * -10
    viz_gt_pre = [[[] for _ in range(nx)] for _ in range(ny)]
    viz_est_pre = [[[] for _ in range(nx)] for _ in range(ny)]

    est_res = test_gain_dyna_full_plain[t]
    log_r_grid = est_res[:, 0:1]
    test_x_est = est_res[:, 1:].reshape((n_test, input_dim - 1))

    xi = gridify(test_gt_x_plain[t, :, args.x_index], args.x_min, args.x_max, nx)
    yi = gridify(test_gt_x_plain[t, :, args.y_index], args.y_min, args.y_max, ny)
    te_xi = xi
    te_yi = yi
    if not args.train_density_only:
        te_xi = gridify(test_x_est[:, args.x_index], args.x_min, args.x_max, nx)
        te_yi = gridify(test_x_est[:, args.y_index], args.y_min, args.y_max, ny)

    data_fwd_cpu_t = test_gtr_plain[t, :, -1]
    r_grid_cpu_t = log_r_grid[:, -1]

    for ii in range(xi.shape[0]):
        if args.viz_in_unlog:
            if args.exp_mode=="kop":
                viz_gt_pre[yi[ii]][xi[ii]].append(np.exp(data_fwd_cpu_t[ii]) * test_ri[ii])
                viz_est_pre[te_yi[ii]][te_xi[ii]].append(np.exp(r_grid_cpu_t[ii]) * test_ri[ii])
            else:
                viz_gt_pre[yi[ii]][xi[ii]].append(np.exp(data_fwd_cpu_t[ii]))
                viz_est_pre[te_yi[ii]][te_xi[ii]].append(np.exp(r_grid_cpu_t[ii]))
        else:
            assert args.exp_mode!="kop"
            viz_gt_pre[yi[ii]][xi[ii]].append(data_fwd_cpu_t[ii])
            viz_est_pre[te_yi[ii]][te_xi[ii]].append(r_grid_cpu_t[ii])
    for ii in range(ny):
        for jj in range(nx):
            if len(viz_gt_pre[ii][jj]) > 0:
                viz_gt[ii, jj] = np.max(viz_gt_pre[ii][jj])
            if len(viz_est_pre[ii][jj]) > 0:
                viz_est[ii, jj] = np.max(viz_est_pre[ii][jj])
    # plot
    if epi == 0:
        plot_heatmap("%s/gt_t%03d.png" % (args.viz_dir, t), viz_gt, args)
    plot_heatmap("%s/est_e%06d_t%03d.png" % (args.viz_dir, epi, t), viz_est, args)
    plot_scatter("%s/all_e%06d_t%03d.png" % (args.viz_dir, epi, t),
                 test_gt_x_plain[t, :, args.x_index], test_gt_x_plain[t, :, args.y_index], "blue",
                 test_x_est[:, args.x_index], test_x_est[:, args.y_index], "red", args)
    return

def parse_line(line, keyword, is_int=False, is_str=False, fail_none=False):
    if fail_none:
        if keyword not in line:
            print("Cannot find %s in cmdline" % keyword)
            return None
    if is_int:
        val = int(line.split(keyword)[1].strip().split(" --")[0])
        print("Parse %d from %s ~~" % (val, keyword))
    elif is_str:
        val = line.split(keyword)[1].strip().split(" --")[0]
        print("Parse %s from %s ~~" % (val, keyword))
    else:
        val = float(line.split(keyword)[1].strip().split(" --")[0])
        print("Parse %.4f from %s ~~"%(val, keyword))
    return val

def main():
    t1=time.time()
    args = get_args()

    assert args.log_density

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # TODO auto load config from cmd.txt
    cmdline = open(os.path.join("cache",args.train_data_path, "cmd.txt")).readlines()[0]
    if args.x_min is None:
        args.x_min = parse_line(cmdline, "--x_min")
    if args.y_min is None:
        args.y_min = parse_line(cmdline, "--y_min")
    if args.x_max is None:
        args.x_max = parse_line(cmdline, "--x_max")
    if args.y_max is None:
        args.y_max = parse_line(cmdline, "--y_max")
    if args.dt is None:
        args.dt = parse_line(cmdline, "--dt")
    if args.max_len is None:
        args.max_len = parse_line(cmdline, "--nt", is_int=True)
    if args.x_index == 0:
        args.x_index = parse_line(cmdline, "--x_index", is_int=True, fail_none=True)
        if args.x_index is None:
            args.x_index = 0
    if args.y_index == 1:
        args.y_index = parse_line(cmdline, "--y_index", is_int=True, fail_none=True)
        if args.y_index is None:
            args.y_index = 1
    if args.x_label == 'x':
        args.x_label = parse_line(cmdline, "--x_label", is_str=True, fail_none=True)
        if args.x_label is None:
            args.x_label = 'x'
    if args.y_label == 'y':
        args.y_label = parse_line(cmdline, "--y_label", is_str=True, fail_none=True)
        if args.y_label is None:
            args.y_label = 'y'
    if args.nx is None:
        args.nx = parse_line(cmdline, "--nx", is_int=True)
    if args.ny is None:
        args.ny = parse_line(cmdline, "--ny", is_int=True)

    # TODO setup exp dir
    args = sub_utils.setup_data_exp_and_logger(args, train_nn=True)
    writer = SummaryWriter(args.exp_dir_full)

    # TODO load data
    data, train_d, test_d, in_means, in_stds, out_means, out_stds, train_ri, test_ri = get_data_tensor_from_numpy(args, "traj_data.npz")
    args.in_means = in_means
    args.in_stds = in_stds
    args.out_means = out_means
    args.out_stds = out_stds
    nt, n_train, ndim = train_d["s"].shape
    n_test = test_d["s"].shape[1]
    input_dim = ndim + 1  # plus t dimension

    # (T*N*3) x,y,t
    test_x0s = test_d["s"][0:1, :, :ndim].repeat(nt, 1, 1)
    test_xt0s = torch.cat([test_x0s, test_d["t"]], dim=-1).reshape((-1, ndim + 1))
    test_gt_xs = test_d["s"].reshape((-1, ndim))
    test_gt_logr = test_d["rho"].reshape(-1, 1)  # data shape (T*NxNy, 1)

    args.ndim = ndim
    args.input_dim = input_dim

    model = Net(args)

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        for k in ["s", "rho", "nabla", "t"]:
            train_d[k] = train_d[k].cuda()
            test_d[k] = test_d[k].cuda()
        test_xt0s = test_xt0s.cuda()
        test_gt_xs = test_gt_xs.cuda()
        test_gt_logr = test_gt_logr.cuda()
        model = model.cuda()
        model.update_device()

    train_gtr = train_d['rho'].detach().cpu()
    test_gtr = test_d['rho'].detach().cpu()
    test_gt_x = test_d['s'].detach().cpu()

    if args.pretrained_path is not None:
        print("Load from %s..."%(args.pretrained_path))
        model.load_state_dict(torch.load(args.pretrained_path))
    optimizer = torch.optim.SGD(model.parameters(), args.lr)

    losses = sub_utils.AverageMeter()
    losses_pde = sub_utils.AverageMeter()
    losses_bound = sub_utils.AverageMeter()
    losses_train_l2 = sub_utils.AverageMeter()
    losses_train_l2_0 = sub_utils.AverageMeter()
    losses_test_l2 = sub_utils.AverageMeter()
    losses_test_l2_0 = sub_utils.AverageMeter()
    losses_train_x_l2 = sub_utils.AverageMeter()
    losses_test_x_l2 = sub_utils.AverageMeter()
    losses_test_x_0_l2 = sub_utils.AverageMeter()

    # TODO prepare for xys for gain mode
    nabla_list = []
    if args.less_t:
        the_t_dim = 1
    else:
        the_t_dim = nt
    for t in range(the_t_dim):
        nabla = train_d['nabla'][t:-1].reshape(-1, 1)
        nabla_list.append(nabla)
    nablas = torch.cat(nabla_list, dim=0)

    # data preparation
    all_xs = train_d['s'].reshape(-1, ndim)
    t_zeros = torch.zeros_like(all_xs[:, 0:1])
    xs_t0 = torch.cat([all_xs, t_zeros], dim=-1)

    train_init_xs = train_d['s'][0].repeat(nt, 1)
    train_init_ts = train_d['t'].reshape(nt * n_train, 1)
    train_init_xts = torch.cat([train_init_xs, train_init_ts], dim=-1)
    pool = Pool(processes=args.num_workers)

    # MAIN LOOP
    for epi in range(args.num_epochs):
        if args.eval_mode == False:
            model.train()

        if args.change_after is not None and epi==args.change_after:
            print("Changed from DYNA to RHO~~")
            args.train_dyna_only=False
            if args.new_show_stat:
                args.show_stat = args.new_show_stat
            if args.new_dyna_weight is not None:
                args.dyna_weight = args.new_dyna_weight
            if args.new_lr is not None:
                args.lr = args.new_lr

            # TODO reset
            losses = sub_utils.AverageMeter()
            losses_pde = sub_utils.AverageMeter()
            losses_bound = sub_utils.AverageMeter()
            losses_train_l2 = sub_utils.AverageMeter()
            losses_train_l2_0 = sub_utils.AverageMeter()
            losses_test_l2 = sub_utils.AverageMeter()
            losses_test_l2_0 = sub_utils.AverageMeter()
            losses_train_x_l2 = sub_utils.AverageMeter()
            losses_test_x_l2 = sub_utils.AverageMeter()
            losses_test_x_0_l2 = sub_utils.AverageMeter()

        # all (x,0) cases, should gain=1  (log(gain)=0)
        gain_dyna_0 = model(xs_t0)
        if args.only_dynamics_arch == False:
            gain_0 = gain_dyna_0[:, 0:1]
            loss_bound = torch.mean(gain_0*gain_0)

        # all (x,t) cases, loss for density and x
        train_est_gain_dyna = model(train_init_xts)
        if args.only_dynamics_arch == False:
            if args.only_density_arch:
                log_train_est_rho = train_est_gain_dyna[:, 0:1]
                pred = train_est_gain_dyna.reshape([nt, n_train, 1])
                gain_dyna_ijks = pred[:-1].reshape([(nt - 1) * n_train, 1])
                gain_dyna_ijk1s = pred[1:].reshape([(nt - 1) * n_train, 1])
            else:
                log_train_est_rho = train_est_gain_dyna[:, 0:1]
                pred = train_est_gain_dyna.reshape([nt, n_train, input_dim])
                gain_dyna_ijks = pred[:-1].reshape([(nt - 1) * n_train, input_dim])
                gain_dyna_ijk1s = pred[1:].reshape([(nt - 1) * n_train, input_dim])
            if args.wrap_log1:
                exp_kk1s = torch.exp(gain_dyna_ijk1s[:,0:1] - gain_dyna_ijks[:,0:1])
                nabla_f_dt1s = nablas * args.dt - 1
                # loss_pde = torch.mean((exp_kk1s + nabla_f_dt1s) ** 2)

                if args.ratio_loss:
                    loss_pde = torch.mean((exp_kk1s / nabla_f_dt1s + 1) ** 2)
                elif args.l1_loss:
                    loss_pde = torch.mean(torch.abs(exp_kk1s + nabla_f_dt1s))
                else:
                    loss_pde = torch.mean((exp_kk1s + nabla_f_dt1s) ** 2)

            elif args.wrap_log2:
                kk1s = gain_dyna_ijk1s[:,0:1] - gain_dyna_ijks[:,0:1]
                log_nabla_f_dt1s = torch.log(nablas * args.dt - 1)
                loss_pde = torch.mean((kk1s + log_nabla_f_dt1s) ** 2)

            log_train_l2_loss = (log_train_est_rho - train_d['rho'].reshape(nt * n_train, 1)) ** 2
            log_train_l2_loss_curve = log_train_l2_loss * 1.0
            log_train_l2_loss_0 = torch.mean(log_train_l2_loss[0])
            log_train_l2_loss = torch.mean(log_train_l2_loss)
            losses_train_l2.update(log_train_l2_loss.detach().cpu().item())
            losses_train_l2_0.update(log_train_l2_loss_0.detach().cpu().item())

        if args.only_density_arch == False:
            if args.only_dynamics_arch:
                if args.normalize:
                    train_x_l2_loss = ((train_est_gain_dyna[:, :] - train_d['s'].reshape(nt * n_train, input_dim - 1))
                                      /model.out_stds[:]
                                      ) ** 2
                else:
                    train_x_l2_loss = (train_est_gain_dyna[:, :] - train_d['s'].reshape(nt * n_train, input_dim-1))**2
            else:
                if args.normalize:
                    train_x_l2_loss = ((train_est_gain_dyna[:, 1:] - train_d['s'].reshape(nt * n_train, input_dim - 1))
                                      /model.out_stds[1:]
                                      ) ** 2
                else:
                    train_x_l2_loss = (train_est_gain_dyna[:, 1:] - train_d['s'].reshape(nt * n_train, input_dim-1))**2

            train_x_0_l2_loss = torch.mean(train_x_l2_loss.reshape(nt, n_train, input_dim-1)[0])
            train_x_l2_loss_curve = train_x_l2_loss * 1.0
            train_x_l2_loss = torch.mean(train_x_l2_loss)
            losses_train_x_l2.update(train_x_l2_loss.detach().cpu().item())


        # writer.add_scalar("loss_bound", loss_bound, epi)
        # writer.add_scalar("loss_pde", loss_pde * args.beta, epi)
        # writer.add_scalar("train_l2_loss", log_train_l2_loss, epi)
        # writer.add_scalar("train_l2_loss_0", log_train_l2_loss_0, epi)

        # BACKPROP
        if args.train_dyna_only:
            loss = args.dyna_weight * train_x_l2_loss
        elif args.train_density_only:
            loss = 1.0 * loss_bound + args.beta * loss_pde
        else:
            loss = 1.0 * loss_bound + args.beta * loss_pde + args.dyna_weight * train_x_l2_loss

        if args.init_x_weight is not None:
            loss = loss + train_x_0_l2_loss * args.init_x_weight

        if args.eval_mode == False:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-6)
            optimizer.step()
            optimizer.zero_grad()

        losses.update(loss.detach().cpu().item())
        if args.only_dynamics_arch==False:
            losses_pde.update(loss_pde.detach().cpu().item())
            losses_bound.update(loss_bound.detach().cpu().item())

        if epi % args.save_freq == 0:
            test_gain_dyna = model(test_xt0s)
            if args.only_density_arch == False and args.only_dynamics_arch == False:
                test_gain_dyna_full = test_gain_dyna.reshape(nt, n_test, input_dim).detach()
            elif args.only_density_arch==True:
                test_gain_dyna_full = test_gain_dyna.reshape(nt, n_test, 1).detach()
            else: # dynamics
                test_gain_dyna_full = test_gain_dyna.reshape(nt, n_test, input_dim-1).detach()
            log_est_rhos = test_gain_dyna[:, 0:1]

            if args.only_density_arch == False:
                if args.only_dynamics_arch == False:
                    if args.normalize:
                        test_x_l2_loss = ((test_gt_xs - test_gain_dyna[:, 1:])/model.out_stds[1:]) ** 2
                    else:
                        test_x_l2_loss = (test_gt_xs - test_gain_dyna[:, 1:]) ** 2
                else:
                    if args.normalize:
                        test_x_l2_loss = ((test_gt_xs - test_gain_dyna[:, :])/model.out_stds[:]) ** 2
                    else:
                        test_x_l2_loss = (test_gt_xs - test_gain_dyna[:, :]) ** 2
                test_x_l2_loss_curve = test_x_l2_loss * 1.0
                test_x_l2_loss = torch.mean(test_x_l2_loss)
                losses_test_x_l2.update(test_x_l2_loss.detach().cpu().item())
                if args.only_dynamics_arch == False:
                    if args.normalize:
                        text_x_0_l2_loss = ((test_gt_xs - test_gain_dyna[:, 1:]).reshape(nt, n_test, input_dim-1)[0]/model.out_stds[1:]) ** 2
                    else:
                        text_x_0_l2_loss = ((test_gt_xs - test_gain_dyna[:, 1:]).reshape(nt, n_test, input_dim-1)[0]) ** 2
                else:
                    if args.normalize:
                        text_x_0_l2_loss = ((test_gt_xs - test_gain_dyna[:, :]).reshape(nt, n_test, input_dim-1)[0]/model.out_stds[:]) ** 2
                    else:
                        text_x_0_l2_loss = ((test_gt_xs - test_gain_dyna[:, :]).reshape(nt, n_test, input_dim-1)[0]) ** 2
                text_x_0_l2_loss = torch.mean(text_x_0_l2_loss)
                losses_test_x_0_l2.update(text_x_0_l2_loss.detach().cpu().item())
            if args.only_dynamics_arch == False:
                log_test_l2_loss = (log_est_rhos - test_gt_logr) ** 2
                if args.more_error:
                    rel_l2_loss = ((log_est_rhos - test_gt_logr) / torch.clamp_min(test_gt_logr, 1e-4))**2
                    print("rel_loss", torch.mean(rel_l2_loss).detach().cpu().item())

                log_test_l2_loss_curve = log_test_l2_loss * 1.0
                log_test_l2_loss_0 = log_test_l2_loss.reshape(nt, n_test)[0]
                log_test_l2_loss = torch.mean(log_test_l2_loss)
                log_test_l2_loss_0 = torch.mean(log_test_l2_loss_0)
                losses_test_l2.update(log_test_l2_loss.detach().cpu().item())
                losses_test_l2_0.update(log_test_l2_loss_0.detach().cpu().item())
                writer.add_scalar("test_l2_loss", log_test_l2_loss, epi)
                writer.add_scalar("test_l2_loss_0", log_test_l2_loss_0, epi)

            if args.train_dyna_only and args.to_see_dens==False:
                print_str = "[%03d/%03d] loss:%.6f(%.6f)" % (
                                epi, args.num_epochs, losses.val, losses.avg,
                            )
            else:
                print_str = "[%03d/%03d] loss:%.6f(%.6f) mse:%.6f(%.6f) pde:%.6f(%.6f) " \
                            "l2_0:%.6f(%.6f) l2:%.6f(%.6f) test:%.6f(%.6f) %.6f(%.6f)" % (
                                epi, args.num_epochs, losses.val, losses.avg,
                                losses_bound.val, losses_bound.avg, losses_pde.val, losses_pde.avg,
                                losses_train_l2_0.val, losses_train_l2_0.avg,
                                losses_train_l2.val, losses_train_l2.avg,
                                losses_test_l2_0.val, losses_test_l2_0.avg,
                                losses_test_l2.val, losses_test_l2.avg,
                            )
            if args.only_density_arch==False:
                print_str += " x2:%.6f(%.6f) te-x2:%.6f(%.6f) t0:%.6f(%.6f)"%\
                             (losses_train_x_l2.val, losses_train_x_l2.avg, losses_test_x_l2.val, losses_test_x_l2.avg,
                              losses_test_x_0_l2.val, losses_test_x_0_l2.avg)
            print(print_str)
            model_path = "%s/model%d.ckpt" % (args.model_dir, epi)
            torch.save(model.state_dict(), model_path)
            if args.only_density_arch == False and args.only_dynamics_arch==False:
                sub_utils.save_model_in_julia_format(
                    model_path, model_path.replace("model%d.ckpt"%(epi), "checkpoint%d.mat"%(epi)), input_dim, input_dim,
                    args
                )
        writer.flush()

        # EVAL AND VIS
        if epi % args.eval_freq == 0 or epi == args.num_epochs - 1:
            if args.only_dynamics_arch == False:
                # show stats
                for t in range(nt):
                    if args.show_stat:
                        if t % 10 == 0 or t == nt - 1 or (nt<=5):
                            est_res = test_gain_dyna_full[t]
                            log_r_grid = est_res[:, 0:1]
                            print("ti %02d min gt:%.4f | est:%.4f || max gt:%.4f | est:%.4f || error:%.4f"%(
                                    t, torch.min(test_gtr[t, :, -1]), torch.min(log_r_grid.detach().cpu()),
                                    torch.max(test_gtr[t, :, -1]), torch.max(log_r_grid.detach().cpu()),
                                torch.mean(((test_gtr[t, :, -1]) - log_r_grid.detach().cpu()[:, -1])**2)
                            ))

            # # loss curve plots
            # plot_loss_curve(nt, train_x_l2_loss_curve, "%s/dyna_train_e%06d.png" % (args.viz_dir, epi))
            # plot_loss_curve(nt, test_x_l2_loss_curve, "%s/dyna_test_e%06d.png" % (args.viz_dir, epi))
            # plot_loss_curve(nt, log_train_l2_loss_curve, "%s/rho_train_e%06d.png" % (args.viz_dir, epi))
            # plot_loss_curve(nt, log_test_l2_loss_curve, "%s/rho_test_e%06d.png" % (args.viz_dir, epi))

            # each time step, density and particles

            tt0 = time.time()
            test_gt_x_plain = test_gt_x.detach().cpu()
            test_gtr_plain = test_gtr.detach().cpu()
            if args.only_density_arch == False and args.only_dynamics_arch == False:
                test_gain_dyna_full_plain = test_gain_dyna_full.detach().cpu()
            elif args.only_density_arch:
                test_gain_dyna_full_plain = torch.cat([test_gain_dyna_full.detach().cpu(), test_gt_x_plain], axis=-1)
            else:
                test_gain_dyna_full_plain = torch.cat([test_gtr_plain, test_gain_dyna_full.detach().cpu()], axis=-1)

            inputs = [[epi, t, test_gain_dyna_full_plain, test_gt_x_plain, test_gtr_plain, n_test, input_dim, test_ri, args] for t in range(nt)]
            pool.map(plot_t_density_particles, inputs)
            tt1 = time.time()
            print("eval+vis time: %.4f sec (%.4f sec/frame)" % (tt1-tt0, (tt1-tt0) / nt))

            if epi==0:
                np.savez(os.path.join("%s/gt%d" % (args.model_dir, epi)),
                        test_gt_logr=test_gt_logr.detach().cpu().numpy(),
                        test_gt_xs=test_gt_xs.detach().cpu().numpy())
                np.savez(os.path.join("%s/stat%d" % (args.model_dir, epi)),
                        in_means=args.in_means,
                        in_stds=args.in_stds,
                        out_means=args.out_means,
                        out_stds=args.out_stds)
            np.savez(os.path.join("%s/pred%d" % (args.model_dir, epi)),
                    test_gain_dyna=test_gain_dyna.detach().cpu().numpy(),
                    log_est_rhos=log_est_rhos.detach().cpu().numpy())

    writer.close()
    t2 = time.time()
    print("Finished in %.4f seconds" % (t2 - t1))


if __name__ == "__main__":
    main()