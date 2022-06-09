import os, sys
from os.path import join as ospj
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sub_utils
import importlib
import hyperparams
from scipy.integrate import odeint
from tqdm import tqdm

import matplotlib
font = {
    # 'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16
}
matplotlib.rc('font', **font)

class MockArgs:
    pass


def main():
    core = importlib.import_module('benchmarks.%s.core'%(args.exp_mode))
    # TODO get controller()
    if args.exp_mode in ["robot", "dint", "toon", "quad", "acc"]:
        mock_args = MockArgs()
    else:
        mock_args = None

    is_gcas = args.exp_mode=="gcas"

    if not is_gcas:
        bench = core.Benchmark(mock_args, args)

        # TODO initial configuration
        np.random.seed(args.random_seed)
        if args.gaussian_init:
            assert len(args.gaussian_mean)==bench.n_dim
            assert len(args.gaussian_vars) == bench.n_dim
            from scipy.stats import multivariate_normal as mm
            if args.special_init:
                pts = np.random.multivariate_normal(mean=args.gaussian_mean, cov=np.diag(args.gaussian_vars) * args.secondary_gain,
                         size=int(args.split_ratio * args.num_samples))
                in_mask = np.where(
                    np.product(np.logical_and(pts < args.s_maxs, pts > args.s_mins), axis=-1),
                )[0]
                pts = pts[in_mask, :]
                while (pts.shape[0] < args.num_samples):
                    sub_pts = np.random.multivariate_normal(mean=args.gaussian_mean, cov=np.diag(args.gaussian_vars), size=args.num_samples)
                    in_mask = np.where(
                        np.product(np.logical_and(sub_pts < args.s_maxs, sub_pts > args.s_mins), axis=-1),
                    )[0]
                    new_pts = sub_pts[in_mask, :]
                    pts = np.concatenate((pts, new_pts), axis=0)
                    if pts.shape[0] > args.num_samples:
                        pts = pts[:args.num_samples]
                state = pts
                init_dens = args.split_ratio * mm.pdf(state, mean=args.gaussian_mean, cov=np.diag(args.gaussian_vars) * args.secondary_gain) + \
                            (1-args.split_ratio)*mm.pdf(state, mean=args.gaussian_mean, cov=np.diag(args.gaussian_vars))
            else:
                state = np.random.multivariate_normal(mean=args.gaussian_mean,
                                                      cov=np.diag(args.gaussian_vars), size=args.num_samples)
                init_dens = mm.pdf(state, mean=args.gaussian_mean, cov=args.gaussian_vars)
        elif args.circle_init:
            assert bench.n_dim==2
            length = np.sqrt(np.random.uniform(0, 1, args.num_samples)) * 2.5
            angle = np.pi * np.random.uniform(0, 2, args.num_samples)
            x = length * np.cos(angle)
            y = length * np.sin(angle)
            state = np.stack((x, y), axis=-1)
        else:
            state = np.random.rand(args.num_samples, bench.n_dim)
            state = state * (np.array(args.s_maxs) - np.array(args.s_mins)) + np.array(args.s_mins)
    else:
        new_dim=13
        if args.gcas_list is not None:
            if ".txt" in args.gcas_list[0]:
                file_list = open(args.gcas_list[0]).readlines()
            else:
                file_list=args.gcas_list
            data={"states":[]}
            sim_nabla_list=[]
            for gcas_path in file_list:
                new_data = np.load(gcas_path+"/sim_trajs.npz", allow_pickle=True)["arr_0"].item()
                the_seed_line = list(open(gcas_path+"/simple.log").readlines())[0]
                print(the_seed_line)
                sim_nabla_list.append(np.stack([[np.sum(np.diagonal(x[:new_dim])) for x in xs] for xs in new_data["grad_list"]]))
                data["states"].append(new_data["states"])
                print("Collect %d trajectories~"%(new_data["states"].shape[0]))
            sim_nabla_list = np.concatenate(sim_nabla_list, axis=0)
            data["states"] = np.concatenate(data["states"], axis=0)
            print("Find %d trajectories from %d files"%(data["states"].shape[0], len(args.gcas_list)))
        else:
            data=np.load(args.sim_data_path, allow_pickle=True)["arr_0"].item()
        print("range")
        for i, key in enumerate(
                ["Vt", "α", "β", "φ", "θ", "ψ", "P", "Q", "R", "Pn", "Pe", "alt", "pow", "Nz", "Ps", "Ny+r"]):
            print("[%02d] %5s   min:%.4f   max:%.4f" % (
            i, key, np.min(data["states"][:, 0, i]), np.max(data["states"][:, 0, i]),))

        strs = ["vt", "alpha", "beta", "phi", "theta", "psi", "p", "q", "r", "pn", "pe", "alt", "power", "nz", "ps",
                "ny"]
        for i, key in enumerate(strs):
            print("%s_min=(%s-%s_mean)/%s_scale" % (strs[i], np.min(data["states"][:, :, i]), strs[i], strs[i]))
            print("%s_max=(%s-%s_mean)/%s_scale" % (strs[i], np.max(data["states"][:, :, i]), strs[i], strs[i]))
        if args.gcas_list is not None:
            args.num_samples = data["states"].shape[0]
        else:
            assert data["states"].shape[0] == args.num_samples
            assert data["states"].shape[1] == args.nt or args.time_gap is not None
            sim_nabla_list = np.stack([[np.sum(np.diagonal(x[:new_dim])) for x in xs] for xs in data["grad_list"]])

    s_list = []
    rho_list = []
    nabla_list = []

    # TODO used for ode-integration
    if args.use_ode:
        rho_list = np.zeros((args.num_samples, args.nt))
        s_list = np.zeros((args.num_samples, args.nt, state.shape[1]))
        if args.gaussian_init:
            rho = init_dens  # np.ones((args.num_samples,))
        else:
            rho = np.ones((args.num_samples,))
        xrho_stack = np.concatenate((state, rho.reshape((-1, 1))), axis=-1)
        for k in tqdm(range(args.num_samples)):
            new_xrho = odeint(bench.get_dx_and_drho, xrho_stack[k], [ti*args.dt for ti in range(args.nt)])
            for ti in range(args.nt):
                rho_list[k, ti] = new_xrho[ti][-1]
                s_list[k, ti, :] = new_xrho[ti][:-1]

            if k in args.buffer_list:
                ts = np.array([tti*args.dt for tti in range(args.nt)])
                tmp_traj_data={"s_list": s_list[:k], "rho_list": rho_list[:k]}
                tmp_traj_data["t_list"] = np.stack(np.tile(ts, (k, 1)))
                np.savez(ospj(args.exp_dir_full, "traj_data_n%d.npz"%(k)), tmp_traj_data)


    # TODO time loop simulation
    for ti in range(args.nt):
        print("Sim t=",ti)
        if args.time_gap is not None and args.time_gap * ti >data["states"].shape[1]-1:
            break
        # time order
        # x_0, rho_0, nabla_0 (k starts from 0)
        # x_k->x_k+1, x_k->nabla_k, (rho_k,nabla_k)->rho_k+1
        # x_k+1 is generated by f(x_k)
        # nabla_k+1 is generated by f(x_k+1)
        # rho_k+1 is derived by rho_k and nabla_k

        if not is_gcas:
            if args.use_ode:
                u, du_cache, new_s = bench.get_u_du_new_s(s_list[:, ti])
            else:
                u, du_cache, new_s = bench.get_u_du_new_s(state)

        if not args.use_ode:
            if ti == 0:
                rho = np.ones((args.num_samples,))
            else:
                drho_dt = -nabla_list[-1] * rho_list[-1]
                rho = rho_list[-1] + drho_dt * args.dt

        # TODO consider the perturbation method!
        if not is_gcas:
            if args.use_ode:
                nabla = bench.get_nabla(s_list[:, ti], u, du_cache)
            else:
                nabla = bench.get_nabla(state, u, du_cache)
        else:
            if args.time_gap is not None:
                nabla = sim_nabla_list[:, ti * (args.time_gap)]
                state = data["states"][:, ti * (args.time_gap), :new_dim]
            else:
                nabla = sim_nabla_list[:, ti]
                state = data["states"][:, ti, :new_dim]

        nabla_list.append(nabla)
        if not args.use_ode:
            s_list.append(state)
            rho_list.append(rho)
            # Move to next cycle
            if not is_gcas:
                state = np.array(new_s)

        # visualization
        if ti % args.viz_freq == 0:
            # particles
            if args.use_ode:
                plt.scatter(s_list[:, ti, args.x_index], s_list[:, ti, args.y_index], s=1)
            else:
                plt.scatter(s_list[ti][:, args.x_index], s_list[ti][:, args.y_index], s=1)
            plt.xlim(args.x_min, args.x_max)
            plt.ylim(args.y_min, args.y_max)
            plt.xlabel(args.x_label)
            plt.ylabel(args.y_label)
            ax = plt.gca()
            ax.set_aspect(1.0 / ((args.y_max - args.y_min) / (args.x_max - args.x_min)), adjustable='box')
            plt.tight_layout()
            plt.savefig(ospj(args.viz_dir, "tmp_%03d.png" % (ti)), bbox_inches='tight', pad_inches=0)
            plt.close()

            # density
            # if args.viz_log_density:
            canvas_log = np.ones((args.ny, args.nx)) * args.viz_log_thres
            # else:
            canvas_norm = np.zeros((args.ny, args.nx))
            viz_gt_pre = [[[] for _ in range(args.nx)] for _ in range(args.ny)]
            if args.use_ode:
                xs = s_list[:, ti, args.x_index]
                ys = s_list[:, ti, args.y_index]
            else:
                xs = s_list[ti][:, args.x_index]
                ys = s_list[ti][:, args.y_index]
            xi = ((xs - args.x_min) / ((args.x_max - args.x_min) / args.nx)).astype(np.int32)
            yi = ((ys - args.y_min) / ((args.y_max - args.y_min) / args.ny)).astype(np.int32)
            xi = np.clip(xi, 0, args.nx - 1)
            yi = np.clip(yi, 0, args.ny - 1)

            for ii in range(xi.shape[0]):
                if args.gaussian_init:
                    if args.use_ode:
                        viz_gt_pre[yi[ii]][xi[ii]].append(rho_list[ii][ti])  # * init_dens[ii])
                    else:
                        viz_gt_pre[yi[ii]][xi[ii]].append(rho_list[ti][ii])  # * init_dens[ii])
                else:
                    if args.use_ode:
                        viz_gt_pre[yi[ii]][xi[ii]].append(rho_list[ii][ti])
                    else:
                        viz_gt_pre[yi[ii]][xi[ii]].append(rho_list[ti][ii])
                # viz_gt_pre[yi[ii]][xi[ii]].append(1.0)
            for ii in range(args.ny):
                for jj in range(args.nx):
                    if len(viz_gt_pre[ii][jj]) > 0:
                        # canvas[ii, jj] = np.max(viz_gt_pre[ii][jj])
                        canvas_log[ii, jj] = np.max(viz_gt_pre[ii][jj])
                        canvas_norm[ii, jj] = np.max(viz_gt_pre[ii][jj])
                        # canvas[ii, jj] = np.sum(viz_gt_pre[ii][jj])
            # if args.viz_log_density:

            for hi, heatmap in enumerate([np.log(canvas_log), canvas_norm]):
                im = plt.imshow(heatmap, origin='lower', cmap=cm.inferno)
                ax = plt.gca()
                if is_gcas:
                    n_yticks = 8
                    n_xticks = 5
                else:
                    n_xticks = n_yticks = 5
                plt.axis("off")
                # ax.set_xticks(np.linspace(0, args.nx, n_xticks))
                # ax.set_xticklabels(["%.3f" % xx for xx in np.linspace(args.x_min, args.x_max, n_xticks)])
                # ax.set_yticks(np.linspace(0, args.ny, n_yticks))
                # ax.set_yticklabels(["%.3f" % xx for xx in np.linspace(args.y_min, args.y_max, n_yticks)])
                # plt.xlabel(args.x_label)
                # plt.ylabel(args.y_label)
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04, format="%.2f")
                plt.savefig(ospj(args.viz_dir, "%s_%03d.png" % ("rho_log" if hi==0 else "rho_norm", ti)), bbox_inches='tight', pad_inches=0.1)
                plt.close()

            # im = plt.imshow(np.log(canvas_log), origin='lower', cmap=cm.inferno)
            # # else:
            # im = plt.imshow(canvas_norm, origin='lower', cmap=cm.inferno)
            # ax = plt.gca()
            # if is_gcas:
            #     n_yticks = 8
            #     n_xticks = 5
            # else:
            #     n_xticks = n_yticks = 5
            #
            # ax.set_xticks(np.linspace(0, args.nx, n_xticks))
            # ax.set_xticklabels(["%.3f" % xx for xx in np.linspace(args.x_min, args.x_max, n_xticks)])
            # ax.set_yticks(np.linspace(0, args.ny, n_yticks))
            # ax.set_yticklabels(["%.3f" % xx for xx in np.linspace(args.y_min, args.y_max, n_yticks)])
            # plt.xlabel(args.x_label)
            # plt.ylabel(args.y_label)
            #
            # cbar = plt.colorbar(im, fraction=0.046, pad=0.04, format="%.3f")
            # plt.savefig(ospj(args.viz_dir,"rho_%03d.png" % (ti)), bbox_inches='tight', pad_inches=0)
            # plt.close()



    # TODO collect data (n_trajs, nt, ndim)
    traj_data = {}
    if args.use_ode:
        traj_data["s_list"] = s_list
        traj_data["rho_list"] = rho_list
    else:
        traj_data["s_list"] = np.stack(s_list, axis=1)
        traj_data["rho_list"] = np.stack(rho_list, axis=1)
    traj_data["nabla_list"] = np.stack(nabla_list, axis=1)
    ts = np.array([tti*args.dt for tti in range(args.nt)])
    traj_data["t_list"] = np.stack(np.tile(ts, (args.num_samples, 1)))
    np.savez(ospj(args.exp_dir_full, "traj_data.npz"), traj_data)


if __name__ == "__main__":
    t0 = time.time()
    args = hyperparams.parse_args()
    args = sub_utils.setup_data_exp_and_logger(args)
    main()
    t1 = time.time()
    print("Finished in %.4f seconds" % (t1 - t0))
