import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_mode', type=str, choices= \
        ['robot', 'dint', 'toon', 'car', 'quad', 'gcas', 'acc', 'pend', 'vdp', 'circ', "kop"])
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--dt', type=float, default=None)
    parser.add_argument('--nt', type=int, default=50)
    parser.add_argument('--sim_steps', type=int, default=1)

    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--s_mins', nargs="+", type=float)
    parser.add_argument('--s_maxs', nargs="+", type=float)

    parser.add_argument('--x_min', type=float, default=-2)
    parser.add_argument('--x_max', type=float, default=2)
    parser.add_argument('--y_min', type=float, default=-2)
    parser.add_argument('--y_max', type=float, default=2)
    parser.add_argument('--nx', type=int, default=81)
    parser.add_argument('--ny', type=int, default=81)

    parser.add_argument('--viz_freq', type=int, default=1)
    parser.add_argument('--viz_log_density', action='store_true')
    parser.add_argument('--viz_log_thres', type=float, default=1e-2)

    parser.add_argument('--x_index', type=int, default=0)
    parser.add_argument('--y_index', type=int, default=1)
    parser.add_argument('--x_label', type=str, default='x')
    parser.add_argument('--y_label', type=str, default='y')
    parser.add_argument('--sim_data_path', type=str, default=None)

    # TODO
    parser.add_argument('--random_seed', type=int, default=1007)
    parser.add_argument('--use_ode', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_workers', type=int, default=None)

    parser.add_argument('--time_gap', type=int, default=None)
    parser.add_argument('--buffer_list', nargs="+", type=int,
                        default=[5, 10, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000])

    parser.add_argument('--gcas_list', nargs="+", type=str, default=None)

    parser.add_argument('--circle_init', action='store_true')

    parser.add_argument('--gaussian_init', action='store_true')
    parser.add_argument('--gaussian_mean', nargs="+", type=float, default=[1.0, 0.0, 0.0])
    parser.add_argument('--gaussian_vars', nargs="+", type=float, default=[0.0625, 0.25, 0.25])
    parser.add_argument('--load_data_from', type=str, default=None)

    parser.add_argument('--special_init', action='store_true')
    parser.add_argument('--split_ratio', type=float, default=0.05)
    parser.add_argument('--secondary_gain', type=float, default=80.0)


    # TODO
    return parser.parse_args()