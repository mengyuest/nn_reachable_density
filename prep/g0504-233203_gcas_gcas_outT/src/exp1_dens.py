import argparse
import os
from os.path import join as ospj
import time
import numpy as np
import torch
from train_nn import Net
import scipy.interpolate
from scipy.interpolate import RegularGridInterpolator
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.io import loadmat

from datetime import datetime
import sys


class Logger(object):
    def __init__(self, path):
        self._terminal = sys.stdout
        self._log = open(path, "w")

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)

    def flush(self):
        pass


def nn_estimate(net, data, init_x, ts, args):
    n_trajs, ndim = data.shape
    ts_2d = ts.reshape((n_trajs, -1))
    x = np.concatenate((init_x, ts_2d), axis=-1)

    if args.exp_mode=="gcas" and n_trajs>10000:
        x_tensor1 = torch.from_numpy(x[:n_trajs//2]).float()
        o_tensor1 = net(x_tensor1)
        basis_log1 = o_tensor1[:, 0:1].detach().cpu().numpy()

        x_tensor2 = torch.from_numpy(x[n_trajs // 2:]).float()
        o_tensor2 = net(x_tensor2)
        basis_log2 = o_tensor2[:, 0:1].detach().cpu().numpy()

        basis_log = np.concatenate((basis_log1, basis_log2), axis=0)
        est_x = np.concatenate((o_tensor1[:, 1:].detach().cpu().numpy(), o_tensor2[:, 1:].detach().cpu().numpy()), axis=0)
    else:
        x_tensor = torch.from_numpy(x).float()
        o_tensor = net(x_tensor)
        basis_log = o_tensor[:, 0:1].detach().cpu().numpy()
        est_x = o_tensor[:, 1:].detach().cpu().numpy()
    return np.exp(basis_log), est_x

def interpolate(train_xs, scores, test_xs, args):
    if args.interp_method=="grid":  # TODO not implemented
        raise NotImplementedError
    elif args.interp_method=="nn":
        interp = scipy.interpolate.NearestNDInterpolator(train_xs, scores[:, 0])
        return interp(test_xs)


def get_grids(xmins, xmaxs, num_grids):
    print(xmins)
    print(xmaxs)
    xvs= np.meshgrid(
        np.linspace(xmins[0], xmaxs[0], num_grids),
        np.linspace(xmins[1], xmaxs[1], num_grids),
        np.linspace(xmins[2], xmaxs[2], num_grids),
        np.linspace(xmins[3], xmaxs[3], num_grids),
    )
    return np.stack([x.flatten() for x in xvs], axis=-1)

def histogram_interp(train_xs, test_xs, args):
    H, edges = np.histogramdd(train_xs, bins=args.bins, density=True)
    xs=[]
    ys=[]

    ndim = len(H.shape)
    if ndim==2:
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                xs.append([
                           (edges[0][i] + edges[0][i + 1]) / 2,
                           (edges[1][j] + edges[1][j + 1]) / 2,
                           ])
                ys.append(H[i][j])
    if ndim==3:
        for i1 in range(H.shape[0]):
            for i2 in range(H.shape[1]):
                for i3 in range(H.shape[2]):
                        xs.append([
                                   (edges[0][i1] + edges[0][i1 + 1]) / 2,
                                   (edges[1][i2] + edges[1][i2 + 1]) / 2,
                                    (edges[2][i3] + edges[2][i3 + 1]) / 2,
                                   ])
                        ys.append(H[i1][i2][i3])
    if ndim==4:
        for i1 in range(H.shape[0]):
            for i2 in range(H.shape[1]):
                for i3 in range(H.shape[2]):
                    for i4 in range(H.shape[3]):
                        xs.append([
                                   (edges[0][i1] + edges[0][i1 + 1]) / 2,
                                   (edges[1][i2] + edges[1][i2 + 1]) / 2,
                                    (edges[2][i3] + edges[2][i3 + 1]) / 2,
                                    (edges[3][i4] + edges[3][i4 + 1]) / 2,
                                   ])
                        ys.append(H[i1][i2][i3][i4])

    if ndim==6:
        for i1 in range(H.shape[0]):
            for i2 in range(H.shape[1]):
                for i3 in range(H.shape[2]):
                    for i4 in range(H.shape[3]):
                        for i5 in range(H.shape[4]):
                            for i6 in range(H.shape[5]):
                                xs.append([
                                           (edges[0][i1] + edges[0][i1 + 1]) / 2,
                                           (edges[1][i2] + edges[1][i2 + 1]) / 2,
                                            (edges[2][i3] + edges[2][i3 + 1]) / 2,
                                            (edges[3][i4] + edges[3][i4 + 1]) / 2,
                                            (edges[4][i5] + edges[4][i5 + 1]) / 2,
                                            (edges[5][i6] + edges[5][i6 + 1]) / 2,
                                           ])
                                ys.append(H[i1][i2][i3][i4][i5][i6])

    if ndim==7:
        for i1 in range(H.shape[0]):
            for i2 in range(H.shape[1]):
                for i3 in range(H.shape[2]):
                    for i4 in range(H.shape[3]):
                        for i5 in range(H.shape[4]):
                            for i6 in range(H.shape[5]):
                                for i7 in range(H.shape[6]):
                                    xs.append([
                                               (edges[0][i1] + edges[0][i1 + 1]) / 2,
                                               (edges[1][i2] + edges[1][i2 + 1]) / 2,
                                                (edges[2][i3] + edges[2][i3 + 1]) / 2,
                                                (edges[3][i4] + edges[3][i4 + 1]) / 2,
                                                (edges[4][i5] + edges[4][i5 + 1]) / 2,
                                                (edges[5][i6] + edges[5][i6 + 1]) / 2,
                                                (edges[6][i7] + edges[6][i7 + 1]) / 2,
                                               ])
                                    ys.append(H[i1][i2][i3][i4][i5][i6][i7])

    if ndim==13:
        for i1 in range(H.shape[0]):
            for i2 in range(H.shape[1]):
                for i3 in range(H.shape[2]):
                    for i4 in range(H.shape[3]):
                        for i5 in range(H.shape[4]):
                            for i6 in range(H.shape[5]):
                                for i7 in range(H.shape[6]):
                                    for i8 in range(H.shape[7]):
                                        for i9 in range(H.shape[8]):
                                            for i10 in range(H.shape[9]):
                                                for i11 in range(H.shape[10]):
                                                    for i12 in range(H.shape[11]):
                                                        for i13 in range(H.shape[12]):
                                                            xs.append([
                                                                       (edges[0][i1] + edges[0][i1 + 1]) / 2,
                                                                       (edges[1][i2] + edges[1][i2 + 1]) / 2,
                                                                        (edges[2][i3] + edges[2][i3 + 1]) / 2,
                                                                        (edges[3][i4] + edges[3][i4 + 1]) / 2,
                                                                        (edges[4][i5] + edges[4][i5 + 1]) / 2,
                                                                        (edges[5][i6] + edges[5][i6 + 1]) / 2,
                                                                        (edges[6][i7] + edges[6][i7 + 1]) / 2,
                                                                        (edges[7][i8] + edges[7][i8 + 1]) / 2,
                                                                        (edges[8][i9] + edges[8][i9 + 1]) / 2,
                                                                        (edges[9][i10] + edges[9][i10 + 1]) / 2,
                                                                        (edges[10][i11] + edges[10][i11 + 1]) / 2,
                                                                        (edges[11][i12] + edges[11][i12 + 1]) / 2,
                                                                        (edges[12][i13] + edges[12][i13 + 1]) / 2,
                                                                       ])
                                                            ys.append(H[i1][i2][i3][i4][i5][i6][i7][i8][i9][i10][i11][i12][i13])

    if ndim==16:
        for i1 in range(H.shape[0]):
            for i2 in range(H.shape[1]):
                for i3 in range(H.shape[2]):
                    for i4 in range(H.shape[3]):
                        for i5 in range(H.shape[4]):
                            for i6 in range(H.shape[5]):
                                for i7 in range(H.shape[6]):
                                    for i8 in range(H.shape[7]):
                                        for i9 in range(H.shape[8]):
                                            for i10 in range(H.shape[9]):
                                                for i11 in range(H.shape[10]):
                                                    for i12 in range(H.shape[11]):
                                                        for i13 in range(H.shape[12]):
                                                            for i14 in range(H.shape[13]):
                                                                for i15 in range(H.shape[14]):
                                                                    for i16 in range(H.shape[15]):
                                                                        xs.append([
                                                                                   (edges[0][i1] + edges[0][i1 + 1]) / 2,
                                                                                   (edges[1][i2] + edges[1][i2 + 1]) / 2,
                                                                                    (edges[2][i3] + edges[2][i3 + 1]) / 2,
                                                                                    (edges[3][i4] + edges[3][i4 + 1]) / 2,
                                                                                    (edges[4][i5] + edges[4][i5 + 1]) / 2,
                                                                                    (edges[5][i6] + edges[5][i6 + 1]) / 2,
                                                                                    (edges[6][i7] + edges[6][i7 + 1]) / 2,
                                                                                    (edges[7][i8] + edges[7][i8 + 1]) / 2,
                                                                                    (edges[8][i9] + edges[8][i9 + 1]) / 2,
                                                                                    (edges[9][i10] + edges[9][i10 + 1]) / 2,
                                                                                    (edges[10][i11] + edges[10][i11 + 1]) / 2,
                                                                                    (edges[11][i12] + edges[11][i12 + 1]) / 2,
                                                                                    (edges[12][i13] + edges[12][i13 + 1]) / 2,
                                                                                    (edges[13][i14] + edges[13][i14 + 1]) / 2,
                                                                                    (edges[14][i15] + edges[14][i15 + 1]) / 2,
                                                                                    (edges[15][i16] + edges[15][i16 + 1]) / 2,
                                                                                   ])
                                                                        ys.append(H[i1][i2][i3][i4][i5][i6][i7][i8][i9][i10][i11][i12][i13][i14][i15][i16])

    from scipy.interpolate import griddata
    grid_z0 = griddata(np.array(xs), np.array(ys), test_xs, method='nearest')
    return grid_z0


def plot_heat(scores, xys, name, ti, args):
    # density
    # if args.viz_log_density:
    #     canvas = np.ones((args.ny, args.nx)) * args.viz_log_thres
    # else:
    canvas = np.zeros((args.ny, args.nx))
    viz_gt_pre = [[[] for _ in range(args.nx)] for _ in range(args.ny)]
    xs = xys[:, args.x_index]
    ys = xys[:, args.y_index]
    xi = ((xs - args.x_min) / ((args.x_max - args.x_min) / args.nx)).astype(np.int32)
    yi = ((ys - args.y_min) / ((args.y_max - args.y_min) / args.ny)).astype(np.int32)
    xi = np.clip(xi, 0, args.nx - 1)
    yi = np.clip(yi, 0, args.ny - 1)

    for ii in range(xi.shape[0]):
        viz_gt_pre[yi[ii]][xi[ii]].append(scores[ii])
        # viz_gt_pre[yi[ii]][xi[ii]].append(1.0)
    for ii in range(args.ny):
        for jj in range(args.nx):
            if len(viz_gt_pre[ii][jj]) > 0:
                canvas[ii, jj] = np.max(viz_gt_pre[ii][jj])
                # canvas[ii, jj] = np.sum(viz_gt_pre[ii][jj])
    im = plt.imshow(canvas, origin='lower', cmap=cm.inferno)
    ax = plt.gca()

    n_yticks = 8
    n_xticks = 5

    ax.set_xticks(np.linspace(0, args.nx, n_xticks))
    ax.set_xticklabels(["%.3f" % xx for xx in np.linspace(args.x_min, args.x_max, n_xticks)])
    ax.set_yticks(np.linspace(0, args.ny, n_yticks))
    ax.set_yticklabels(["%.3f" % xx for xx in np.linspace(args.y_min, args.y_max, n_yticks)])
    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, format="%.3f")
    plt.savefig(ospj(args.viz_dir, "rho_%s_%03d.png" % (name, ti)), bbox_inches='tight', pad_inches=0)
    plt.close()

def main(args):
    np.random.seed(args.random_seed)
    # pick the "ground-truth" data
    print("Exp-%s"%(args.exp_mode))

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    import socket
    host_name = socket.gethostname()

    traj_data=np.load(args.train_data_path, allow_pickle=True)['arr_0'].item()
    traj_data["rho_list"] = traj_data["rho_list"][:args.max_n_trajs]
    traj_data["s_list"] = traj_data["s_list"][:args.max_n_trajs]
    traj_data["t_list"] = traj_data["t_list"][:args.max_n_trajs]

    # print(traj_data["s_list"].shape, traj_data["t_list"].shape, traj_data["rho_list"].shape)
    # train_xs = traj_data['s_list']
    # train_ts = traj_data['t_list']
    # train_rhos = traj_data["rho_list"]


    # pick a subset (initial condition) for point interpolation
    split = int(args.train_ratio * traj_data['s_list'].shape[0])


    # pick the network estimation
    model_dir = os.path.dirname(args.model_path)
    nn_args = np.load(ospj(model_dir, "..", "%s_args.npz"%(args.exp_mode)), allow_pickle=True)['args'].item()
    nn_args.input_dim = traj_data['s_list'].shape[-1] + 1
    nn_args.only_density_arch = False
    nn_args.only_dynamics_arch = False
    if nn_args.normalize:
        dd=loadmat(os.path.join(os.path.dirname(args.model_path), "..", "%s_checkpoint0.mat"%(args.exp_mode)))
        nn_args.in_means = dd["X_mean"]
        nn_args.out_means = dd["Y_mean"]
        nn_args.in_stds = dd["X_std"]
        nn_args.out_stds = dd["Y_std"]

    net = Net(nn_args)
    net.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

    # measures
    # TODO
    if args.gaussian:
        initial_rho_train = traj_data['rho_list'][:split, 0]
        initial_rho_test = traj_data['rho_list'][split:, 0]
    else:
        if "toon" in args.exp_mode:
            initial_rho = 1.0
        else:
            initial_rho = 1 / np.product(np.array(args.s_maxs) - np.array(args.s_mins))
    res_dir="cache/dens_%s/g%s"%(args.exp_mode, datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S"))
    viz_dir=ospj(res_dir,"viz")
    args.viz_dir = viz_dir
    os.makedirs(viz_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(args.viz_dir, "./simple.log"))
    # print("python " + " ".join(sys.argv))

    score_list=[]
    kl_list=[]
    test_xs_list=[]
    # args.nt = nn_args.max_len
    if args.use_full_span:
        args.nt = traj_data["s_list"].shape[1]
        args.tgap = 1
    else:
        if traj_data["s_list"].shape[1]<20:
            args.nt = traj_data["s_list"].shape[1]
            args.tgap = 1
        else:
            args.nt = 50
            args.tgap = 5
    tspan = range(0, args.nt, args.tgap)

    for ti in tspan:
        train_init_xs = traj_data["s_list"][:split, 0, :]
        train_xs = traj_data['s_list'][:split, ti]
        train_ts = traj_data['t_list'][:split, ti]
        train_rhos = traj_data['rho_list'][:split, ti]
        test_xs = traj_data['s_list'][split:, ti]
        test_rhos = traj_data['rho_list'][split:, ti]

        # train_init_xs = traj_data["s_list"][:, 0, :]
        # train_xs = traj_data['s_list'][:, ti]
        # train_ts = traj_data['t_list'][:, ti]
        # train_rhos = traj_data['rho_list'][:, ti]
        # test_xs = traj_data['s_list'][:, ti]
        # test_rhos = traj_data['rho_list'][:, ti]

        # test_ts = traj_data['t_list'][split:, ti]
        # x_mins = np.min(train_xs, axis=(1))
        # x_maxs = np.max(train_xs, axis=(1))
        # test_xs = get_grids(x_mins, x_maxs, num_grids)

        # print(ti)
        kl_list.append({})
        score_list.append({})
        nn_scores_basis, est_x = nn_estimate(net, train_xs, train_init_xs, train_ts, args) # on train data
        nn_scores = interpolate(train_xs, nn_scores_basis, est_x if args.use_est_x else test_xs, args) # on test points, using interpolation
        nn_scores = np.clip(nn_scores, 1e-36, 1e36)
        if args.gaussian:
            nn_scores = initial_rho_test * nn_scores
        else:
            nn_scores = initial_rho * nn_scores
        # print(nn_scores)
        plot_heat(nn_scores, test_xs , "nn", ti, args)

        # pick the histogram
        if args.skip_histogram:
            hist_scores = np.ones_like(nn_scores)
        else:
            if args.real_histogram:
                hist_scores = histogram_interp(train_xs, test_xs, args)
            else:
                hist = KernelDensity(kernel='tophat', bandwidth=args.tophat_bandwidth).fit(train_xs)
                hist_scores = np.exp(hist.score_samples(test_xs))
            # print(hist_scores)
            plot_heat(hist_scores, test_xs, "hist", ti, args)

        # pick the kernel based
        kde = KernelDensity(kernel='epanechnikov', bandwidth=args.ep_bandwidth).fit(train_xs)
        kde_scores = np.exp(kde.score_samples(test_xs))
        # print(kde_scores)
        plot_heat(kde_scores, test_xs, "kde", ti, args)

        # ground-truth density
        # TODO use ode methods
        # liou_scores = interpolate(train_xs, train_rhos.reshape((-1,1)), test_xs, args)
        # liou_scores = initial_rho * liou_scores
        if args.gaussian:
            liou_scores = test_rhos
        else:
            liou_scores = initial_rho * test_rhos
        plot_heat(liou_scores, test_xs, "gt", ti, args)

        score_list[-1]["nn"] = nn_scores
        score_list[-1]["kde"] = kde_scores
        score_list[-1]["hist"] = hist_scores
        score_list[-1]["liou"] = liou_scores

        # TODO
        # measure the KL divergence
        kl_list[-1]["nn_hist"] = scipy.stats.entropy(nn_scores, hist_scores)
        kl_list[-1]["kde_hist"] = scipy.stats.entropy(kde_scores, hist_scores)
        kl_list[-1]["liou_hist"] = scipy.stats.entropy(liou_scores, hist_scores)

        kl_list[-1]["nn_liou"] = scipy.stats.entropy(nn_scores, liou_scores)
        kl_list[-1]["kde_liou"] = scipy.stats.entropy(kde_scores, liou_scores)
        kl_list[-1]["hist_liou"] = scipy.stats.entropy(hist_scores, liou_scores)
        # print(kl_list[-1])

        test_xs_list.append(test_xs)

    # save data
    np.savez(ospj(res_dir, "result.npz"), score_list=score_list, kl_list=kl_list, test_xs_list=test_xs_list)

    # multiple-process
    # plot?
    colors = [
        # "#417A68",
        "#92AEBB",
        "#E9CEB8",
        "#E7452E",  # RED
    ]

    for cmp in ["hist", "liou"]:
        for key_i, key in enumerate(kl_list[0]):
            if key.endswith(cmp):
                plt.plot(tspan, [x[key] for x in kl_list], label="KL("+key.replace("_", "|")+")", c=colors[key_i%3])
        plt.legend()
        plt.savefig(ospj(viz_dir, "kl_curve%s.png"%(cmp)), bbox_inches='tight', pad_inches=0)
        plt.close()

        # bar plot
        xlabels = ["nn", "kde", "hist", "liou"]
        xlabels.remove(cmp)
        x_ticks = np.arange(len(xlabels))  # the label locations
        width = 0.19
        margin = 0.02

        fig, ax = plt.subplots()
        # ax.set_ylim(0.0006, 0.30)
        # ax.set_ylim(0.88, 1.0)
        # ax.set_yscale('log')
        rects1 = ax.bar(0.0, np.mean([x[xlabels[0]+"_"+cmp] for x in kl_list]), width, color=colors[0])
        rects2 = ax.bar(1.0, np.mean([x[xlabels[1]+"_"+cmp] for x in kl_list]), width, color=colors[1])
        rects3 = ax.bar(2.0, np.mean([x[xlabels[2]+"_"+cmp] for x in kl_list]), width, color=colors[2])
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("Approaches")
        ax.set_ylabel("KL Divergence")
        plt.savefig(ospj(viz_dir, "kl_all_%s.png"%(cmp)), bbox_inches='tight', pad_inches=0)
        plt.close()

        if cmp=="liou":
            print("Vs %s  || NN:%.6f KDE:%.6f Hist:%.6f"%(
                cmp,
                np.mean([x["nn_%s"%cmp] for x in kl_list]),
                np.mean([x["kde_%s"%cmp] for x in kl_list]),
                np.mean([x["hist_%s"%cmp] for x in kl_list])
            ))


if __name__ == "__main__":
    t1=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--exp_mode', type=str, choices= \
        ['robot', 'dint', 'toon', 'car', 'quad', 'gcas', 'acc', 'pend', 'vdp', 'circ', 'kop'])
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--train_data_path', type=str, default=None)
    parser.add_argument('--test_data_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--interp_method', type=str, default='nn')
    parser.add_argument('--random_seed', type=int, default=1007)
    parser.add_argument('--train_ratio', type=float, default=0.3)
    parser.add_argument('--tophat_bandwidth', type=float, default=1.0)
    parser.add_argument('--ep_bandwidth', type=float, default=1.0)
    parser.add_argument('--s_mins', nargs="+", type=float)
    parser.add_argument('--s_maxs', nargs="+", type=float)
    parser.add_argument('--use_est_x', action="store_true")
    parser.add_argument('--x_label', type=str, default="x")
    parser.add_argument('--y_label', type=str, default="y")
    parser.add_argument('--x_index', type=int, default=0)
    parser.add_argument('--y_index', type=int, default=1)
    parser.add_argument('--nx', type=int, default=161)
    parser.add_argument('--ny', type=int, default=161)

    parser.add_argument('--real_histogram', action='store_true')
    parser.add_argument('--bins', type=int, default=25)

    parser.add_argument('--x_min', type=float, default=-4.0)
    parser.add_argument('--x_max', type=float, default=4.0)
    parser.add_argument('--y_min', type=float, default=-4.0)
    parser.add_argument('--y_max', type=float, default=4.0)

    parser.add_argument('--max_n_trajs', type=int, default=99999999)
    parser.add_argument('--skip_histogram', action="store_true")

    parser.add_argument("--gaussian", action="store_true")

    parser.add_argument('--use_full_span', action='store_true')

    args = parser.parse_args()
    main(args)
    t2=time.time()
    print("Finished in %.4f seconds"%(t2-t1))
