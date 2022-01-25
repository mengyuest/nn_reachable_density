import os, sys
import time
import numpy as np
import argparse
import sub_utils
from multiprocessing.pool import Pool
import json
from pypoman import plot_polygon, project_polytope
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_prefix", type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--exp_mode', type=str, choices= \
        ['robot', 'dint', 'toon', 'car', 'quad', 'gcas', 'acc', 'pend', 'vdp', 'circ', 'kop'])
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--output_name', type=str, default="bag.npz")
    parser.add_argument('--nt', type=int, default=None)
    parser.add_argument('--dt', type=float, default=None)
    parser.add_argument('--x_index', type=int, default=0)
    parser.add_argument('--y_index', type=int, default=1)
    return parser.parse_args()


def load_json_files(args):
    data = {}
    with open(args.data_prefix % ("input"), "r") as f:
        data["s2i"] = json.load(f)
    with open(args.data_prefix % ("output"), "r") as f:
        data["s2o"] = json.load(f)
    with open(args.data_prefix % ("backward"), "r") as f:
        data["s2b"] = json.load(f)
    with open(args.data_prefix % ("map"), "r") as f:
        data["s2m"] = json.load(f)
    return data


def timify_poly_repre(input_data):
    ti, i, rpm_tuple, args = input_data
    rpm_d_list = []
    rpm_d = {}
    A, b, C, C_inv, d, E, f = \
        rpm_tuple["A"], rpm_tuple["b"], rpm_tuple["C"], rpm_tuple["C_inv"], \
        rpm_tuple["d"], rpm_tuple["E"], rpm_tuple["f"]
    P_t = np.zeros((2, A.shape[1]))
    P_t[0, -1] = -1
    P_t[1, -1] = 1
    q_t = np.array([-ti * args.dt, ti * args.dt])

    A_t = np.concatenate((A, P_t), axis=0)
    b_t = np.concatenate((b, q_t), axis=0)
    E_t = np.dot(A_t, C_inv)
    f_t = b_t + np.dot(E_t, d)

    rpm_d["C"] = C
    rpm_d["d"] = d
    rpm_d["E"] = E_t
    rpm_d["f"] = f_t

    rpm_d["v_i_full"] = sub_utils.get_A_b_vertices_robust(A_t, b_t, ti, i)
    if rpm_d["v_i_full"] is not None:
        rpm_d["v_i"], rpm_d["vol_i"], status = sub_utils.get_A_b_sys_vertices_volume_robust(rpm_d["v_i_full"],
                                                                                        is_input=True)
    else:
        rpm_d["v_i"] = rpm_d["vol_i"] = None
    rpm_d["v_o_full"] = sub_utils.get_A_b_vertices_robust(E_t, f_t, ti, i)
    if rpm_d["v_o_full"] is not None:
        rpm_d["v_o"], rpm_d["vol_o"], status = sub_utils.get_A_b_sys_vertices_volume_robust(rpm_d["v_o_full"],
                                                                                        is_input=False)
    else:
        rpm_d["v_o"] = rpm_d["vol_o"] = None

    if rpm_d["v_o"] is not None:
        rpm_d["v_o_2"] = [(x[args.x_index], x[args.y_index]) for x in rpm_d["v_o"]]
        _, rhomin, rhomax = sub_utils.non_empty(E_t, f_t)
        rpm_d["g_min"] = rhomin * ti * args.dt
        rpm_d["g_max"] = rhomax * ti * args.dt
        rpm_d["x_min"] = np.min(rpm_d["v_o_full"], axis=0)[1:]
        rpm_d["x_max"] = np.max(rpm_d["v_o_full"], axis=0)[1:]
    else:
        rpm_d["v_o_2"] = None
        rpm_d["g_min"] = None
        rpm_d["g_max"] = None
        rpm_d["x_min"] = None
        rpm_d["x_max"] = None

    rpm_d["ti"] = ti
    rpm_d["i"] = i
    rpm_d["index"] = 0  # TODO(in case we further refine)

    rpm_d_list.append(rpm_d)

    return rpm_d_list


def main():
    t1 = time.time()
    color_list=np.random.rand(5000, 3)
    args = get_args()
    data = load_json_files(args)
    num_states = len(data["s2m"]["A"])
    rpm_tuple_list = []
    for i in range(num_states):
        rpm_tuple = {}
        rpm_tuple["A"] = np.array(data["s2i"]["A"][i]).T
        rpm_tuple["b"] = np.array(data["s2i"]["b"][i]).T
        rpm_tuple["C"] = np.array(data["s2m"]["A"][i]).T
        rpm_tuple["C_inv"] = np.linalg.inv(rpm_tuple["C"])
        rpm_tuple["d"] = np.array(data["s2m"]["b"][i]).T
        rpm_tuple["E"] = np.array(data["s2o"]["A"][i]).T
        rpm_tuple["f"] = np.array(data["s2o"]["b"][i]).T
        rpm_tuple_list.append(rpm_tuple)

    #     #TODO(debug)
    #     v_i_full = sub_utils.get_A_b_vertices_robust(rpm_tuple["A"], rpm_tuple["b"], 0, i)
    #     if v_i_full is not None and len(v_i_full)>=3:
    #         print(v_i_full)
    #         v_i, vol_i, status = sub_utils.get_A_b_sys_vertices_volume_robust(v_i_full, is_input=True)
    #         if v_i is None:
    #             print(v_i, status)
    #             exit()
    #         if v_i is not None:
    #             plot_polygon(v_i, color=color_list[i])
    # plt.xlim(-0.0, 2.0)
    # plt.ylim(-0.0, 2.0)
    # plt.axis("scaled")
    # plt.show()


    # timify polyhedron
    bag = []
    pool = Pool(processes=args.num_workers)

    for ti in range(args.nt):
        input_list = []
        print(ti)
        for i in range(num_states):
            input_list.append((ti, i, rpm_tuple_list[i], args))
        outputs = pool.map(timify_poly_repre, input_list)
        outputs_flatten = []
        for oo in outputs:
            for pack in oo:
                if pack['v_i'] is not None:
                    outputs_flatten.append(pack)
        bag.append(outputs_flatten)

    # for i,pack in enumerate(bag[1]):
    #     plot_polygon(pack["v_i"], color=color_list[0])
    # plt.xlim(-0.0, 2.0)
    # plt.ylim(-0.0, 2.0)
    # plt.axis("scaled")
    # plt.show()

    # save bag to npz
    os.makedirs(args.output_path, exist_ok=True)
    np.savez(os.path.join(args.output_path, args.output_name), bag)
    print("Prep %d states, results %d repres in %.4f s" % (num_states, np.sum([len(b) for b in bag]), time.time() - t1))


if __name__ == "__main__":
    main()
