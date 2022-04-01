import os, sys, time
import numpy as np
import argparse
import sub_utils
import scipy.io

def main():
    old_args = np.load("data/%s_args.npz"%(args.exp_mode), allow_pickle=True)['args'].item()

    '''
    dint_args.npz 10 1.0 0 1
    gcas_args.npz 106 0.03333 0 11
    robot_args.npz 50 0.05 0 1
    car_args.npz 50 0.1 0 1
    kop_args.npz 80 0.125 0 1
    vdp_args.npz 50 0.05 0 1
    quad_args.npz 12 0.1 0 1
    pend_args.npz 50 0.02 0 1
    toon_args.npz 50 0.15 0 1
    acc_args.npz 50 0.1 0 3
    '''
    configs = {
        "acc": {"dim": 8,
                "output_path": "data/",
                "output_name": "%s_bag.npz" % (args.exp_mode),
                "nt": 50,
                "dt": 0.1,
                "x_index": 0,
                "y_index": 3,
                },
        "car": {"dim": 6,
               "output_path": "data/",
               "output_name": "%s_bag.npz"%(args.exp_mode),
               "nt": 50,
               "dt": 0.1,
               "x_index": 0,
               "y_index": 1,
               },
        "dint": {"dim": 3,
               "output_path": "data/",
               "output_name": "%s_bag.npz"%(args.exp_mode),
               "nt": 10,
               "dt": 1.0,
               "x_index": 0,
               "y_index": 1,
               },
        "gcas": {"dim": 14,
                 "output_path": "data/",
                 "output_name": "%s_bag.npz" % (args.exp_mode),
                 "nt": 106,
                 "dt": 0.03333,
                 "x_index": 0,
                 "y_index": 11,
                 },
        "kop": {"dim": 4,
                 "output_path": "data/",
                 "output_name": "%s_bag.npz" % (args.exp_mode),
                 "nt": 80,
                 "dt": 0.125,
                 "x_index": 0,
                 "y_index": 1,
                 },
        "pend": {"dim": 5,
                "output_path": "data/",
                "output_name": "%s_bag.npz" % (args.exp_mode),
                "nt": 50,
                "dt": 0.02,
                "x_index": 0,
                "y_index": 1,
                },
        "quad": {"dim": 7,
                 "output_path": "data/",
                 "output_name": "%s_bag.npz" % (args.exp_mode),
                 "nt": 12,
                 "dt": 0.1,
                 "x_index": 0,
                 "y_index": 1,
                 },
        "robot": {"dim": 5,
                 "output_path": "data/",
                 "output_name": "%s_bag.npz" % (args.exp_mode),
                 "nt": 50,
                 "dt": 0.05,
                 "x_index": 0,
                 "y_index": 1,
                 },
        "toon": {"dim": 17,
                 "output_path": "data/",
                 "output_name": "%s_bag.npz" % (args.exp_mode),
                 "nt": 50,
                 "dt": 0.15,
                 "x_index": 0,
                 "y_index": 1,
                 },
        "vdp": {"dim": 3,
                 "output_path": "data/",
                 "output_name": "%s_bag.npz" % (args.exp_mode),
                 "nt": 50,
                 "dt": 0.05,
                 "x_index": 0,
                 "y_index": 1,
                 },
    }

    cfg = configs[args.exp_mode]

    if args.exp_mode in ["gcas", "acc"]:
        mat = scipy.io.loadmat("./data/%s_checkpoint0.mat"%(args.exp_mode))
        args.in_means = mat["X_mean"]
        args.in_stds = mat["X_std"]
        args.out_means = mat["Y_mean"]
        args.out_stds = mat["Y_std"]

    sub_utils.save_model_in_julia_format(
        model_path="data/models/%s_model.ckpt"%(args.exp_mode),
        save_path="data/models/%s_model.mat"%(args.exp_mode),
        in_dim=cfg["dim"],
        out_dim=cfg["dim"],
        args=old_args)

    ## command line
    ## julia --project=. formal_collect.jl "/home/meng/exps_pde/g0623-105119_vdp_lr5k_e500k_newT_pret/models/checkpoint499000.mat" "vdp" "new2_"
    ## formal_collect.jl
    os.system("julia --project=. formal_collect.jl \"data/models/%s_model.mat\" \"%s\" \"corl\"" % (
        args.exp_mode, args.exp_mode))


    ## handle to bag data
    os.system("python poly_preproc.py --data_prefix data/models/exp_%s_corlstate2%s.json "
              "--exp_mode %s --output_path data/models --output_name %s_bag.npz "
              "--nt %d --dt %.4f --x_index %d --y_index %d" % (
        args.exp_mode, "%s", args.exp_mode, args.exp_mode, cfg["nt"], cfg["dt"], cfg["x_index"], cfg["y_index"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("handle RPM")
    parser.add_argument('--exp_mode', type=str, default=None)
    args = parser.parse_args()
    main()