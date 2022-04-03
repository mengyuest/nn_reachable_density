import os, sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sub_utils
from train_nn import Net, get_data_tensor_from_numpy, get_args, parse_line


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

    # TODO(debug)

    est_out = model(test_xt0s)
    np.savez("%s/test_data.npz"%(args.exp_dir_full),
             x=test_xt0s.detach().cpu().numpy(),
             y=torch.cat([test_gtr, test_gt_x], dim=-1).detach().cpu().numpy(),
             y_est=est_out.detach().cpu().numpy())

    tmp_list=[]
    sim_list=[np.array([1.0,1.0])]
    for i in range(50):
        print(i)

        dx0=sim_list[-1][0]
        dx1=1.0 * (1-sim_list[-1][0]**2)*sim_list[-1][1] - sim_list[-1][0]
        dt = 0.05

        new_x0 = sim_list[-1][0] + dx0 * dt
        new_x1 = sim_list[-1][1] + dx1 * dt
        sim_list.append(np.array([new_x0, new_x1]))

        dbg_input = torch.zeros_like(test_xt0s[0:1, :])
        dbg_input[:, 0] = 1
        dbg_input[:, 1] = 1
        dbg_input[:, 2] = dt * i
        dbg_output = model(dbg_input)
        tmp_list.append(dbg_output.detach().cpu().numpy())
    sim_list = np.stack(sim_list, axis=1)
    tmp_list = np.stack(tmp_list, axis=1)
    plt.plot(sim_list[:, 0], sim_list[:, 1],label="sim", color="blue")
    plt.plot(tmp_list[0,:,1], tmp_list[0,:,2],label="nn", color="red")
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/debug.png"%(args.exp_dir_full))
    plt.close()
    exit()
    # TODO(end of debug)

    t2 = time.time()
    print("Finished in %.4f seconds" % (t2 - t1))


if __name__ == "__main__":
    main()
