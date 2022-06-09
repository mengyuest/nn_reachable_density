import os, sys
from os.path import join as ospj
import time
import numpy as np
import argparse
import sub_utils
from multiprocessing.pool import Pool
import matplotlib
import matplotlib.pyplot as plt
import pylab
from pypoman import plot_polygon, project_polytope
import scipy
from sklearn.neighbors import KernelDensity
from scipy.spatial import ConvexHull
import matplotlib.patches as patches
from scipy.spatial import Delaunay


def draw_background():
    ax = pylab.gca()
    patch = pylab.Polygon([[args.x_min, args.y_min],
                           [args.x_max, args.y_min],
                           [args.x_max, args.y_max],
                           [args.x_min, args.y_max],
                           [args.x_min, args.y_min]], alpha=1.0, color="black",
                          linestyle="solid", fill=True, linewidth=None)
    ax.add_patch(patch)

def update_bag(bag):
    new_bag=[]
    for ti in range(len(bag)):
        new_bag.append([])
        for j in range(len(bag[ti])):
            if bag[ti][j]['v_i'] is not None:
                if bag[ti][j]["vol_o"] is not None:
                    new_bag[ti].append(bag[ti][j])
    return new_bag

def plot_density_patches(rho_list, vertices_list, r_min, r_max, is_rho_min, img_path):
    cmap = matplotlib.cm.inferno
    draw_background()

    if is_rho_min:
        ### viz_queue = sorted(viz_queue, key=lambda x: x[-2], reverse=True)
        # viz_queue = sorted(viz_queue, key=lambda x: x[-2], reverse=False)
        index = np.argsort(rho_list)
    else:
        # viz_queue = sorted(viz_queue, key=lambda x: x[-1], reverse=False)
        index = np.argsort(rho_list)

    for ii in index:
        # vertices, rho_min, rho_max = res
        the_rho = rho_list[ii]
        vertices = vertices_list[ii]
        norm_rho = (the_rho - r_min) / (np.clip(r_max - r_min, a_min=1e-4, a_max=1e10))
        color_i = cmap(norm_rho)
        color_i = (color_i[0], color_i[1], color_i[2], 1.0)
        plot_polygon(vertices, color=color_i, alpha=color_i[-1])

    plt.xlim(args.x_min, args.x_max)
    plt.ylim(args.y_min, args.y_max)
    ax = plt.gca()
    ax.set_aspect(1.0 / ((args.y_max-args.y_min)/(args.x_max-args.x_min)), adjustable='box')
    norm = matplotlib.colors.Normalize(vmin=r_min, vmax=r_max)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
    plt.savefig(img_path)
    plt.close()

def clean_bag(bag, nt):
    for ti in range(nt):
        original_num = len(bag[ti])
        del_indices=[]
        for od_i in range(len(bag[ti])):
            if bag[ti][od_i]["vol_i"] is None:
                del_indices.append(od_i)
        if len(del_indices)>0:
            print("Found %d/%d to remove from bag"%(len(del_indices), original_num))
            for del_i in del_indices:
                del bag[ti][del_i]
    return bag

def prepare_bag_data(bag):
    saved_data = {}


    # TODO clean None items
    bag = clean_bag(bag, args.nt)

    for ti in range(0, args.nt):
        saved_data[ti]={}
        densities=[]
        probabilities=[]
        od_i_list=[]
        # assign density
        for od_i in range(len(bag[ti])):
            od_i_list.append(od_i)
            if len(bag[ti][od_i]["v_i"])==0:
                print(bag[ti][od_i]["v_i"],bag[ti][od_i]["v_o"])
            center = np.mean(bag[ti][od_i]["v_i"], axis=0)
            if args.distribution == "gaussian":
                est_density = sub_utils.gaussian_pdf(center, np.array(args.mu), np.diag(args.cov))
            elif args.distribution == "uniform":
                est_density = 1.0
            densities.append(est_density)
            probabilities.append(est_density * bag[ti][od_i]["vol_i"])

        # normalize it
        densities = np.array(densities)
        probabilities = np.array(probabilities)
        factor = np.sum(probabilities)
        densities /= factor
        probabilities /= factor
        for i, od_i in enumerate(od_i_list):
            bag[ti][od_i]["init_dens"] = densities[i]

        rho_min_list=[]
        rho_max_list=[]
        volume_list=[]
        prob_min_list=[]
        prob_max_list=[]
        vertice_list=[]

        # compute density
        for od_i in range(len(bag[ti])):
            bag[ti][od_i]["rho_min"] = np.exp(bag[ti][od_i]["g_min"]) * bag[ti][od_i]["init_dens"]
            rho_min_list.append(bag[ti][od_i]["rho_min"])
            bag[ti][od_i]["rho_max"] = np.exp(bag[ti][od_i]["g_max"]) * bag[ti][od_i]["init_dens"]
            rho_max_list.append(bag[ti][od_i]["rho_max"])
            volume_list.append(bag[ti][od_i]["vol_o"])
            vertice_list.append(bag[ti][od_i]["v_o_2"])

        rho_max_list = np.array(rho_max_list)
        rho_min_list = np.array(rho_min_list)
        volume_list = np.array(volume_list)
        rv_maxs=rho_max_list*volume_list
        rv_mins=rho_min_list*volume_list

        # compute probability
        for od_i in range(len(bag[ti])):
            bag[ti][od_i]["prob_min"] = rv_mins[od_i] / (np.sum(rv_maxs) - rv_maxs[od_i] + rv_mins[od_i])
            bag[ti][od_i]["prob_max"] = rv_maxs[od_i] / (np.sum(rv_mins) - rv_mins[od_i] + rv_maxs[od_i])
            prob_min_list.append(bag[ti][od_i]["prob_min"])
            prob_max_list.append(bag[ti][od_i]["prob_max"])
        prob_min_list = np.array(prob_min_list)
        prob_max_list = np.array(prob_max_list)

        saved_data[ti]["dens_max_list"] = rho_max_list
        saved_data[ti]["dens_min_list"] = rho_min_list
        saved_data[ti]["volume_list"] = volume_list
        saved_data[ti]["vertice_list"] = vertice_list  # TODO(saved data)
        saved_data[ti]["prob_min_list"] = prob_min_list  # TODO(saved data)
        saved_data[ti]["prob_max_list"] = prob_max_list  # TODO(saved data)
    return bag, saved_data

def plot_frs_func(saved_data):
    pool = Pool(processes=args.num_workers)
    input = [(ti, saved_data[ti]) for ti in range(0, args.nt)]
    pool.map(plot_frs_func_worker, input)
    pool.close()

def plot_frs_func_worker(input_data):
    ti, data = input_data
    print(ti)
    rho_max_list = data["dens_max_list"]
    rho_min_list = data["dens_min_list"]
    vertice_list = data["vertice_list"]
    prob_min_list = data["prob_min_list"]
    prob_max_list = data["prob_max_list"]

    # plot frs-density
    if args.plot_frs_density:
        rho_minimum = np.min(rho_min_list)
        rho_maximum = np.max(rho_max_list)
        if args.dens_log_val:
            rho_min_clip = args.dens_log_val_minimum
            rho_minimum = np.log(rho_minimum)
            rho_maximum = np.log(rho_maximum)
            rho_max_list = np.log(rho_max_list)
            rho_min_list = np.log(rho_min_list)
        else:
            rho_min_clip = 0
        r_min = min(rho_min_clip, rho_minimum)
        r_max = max(rho_min_clip, rho_maximum)

        plot_density_patches(rho_min_list, vertice_list, r_min, r_max, is_rho_min=True,
                             img_path="%s/rho_min_t%02d.png" % (args.viz_dir, ti))
        plot_density_patches(rho_max_list, vertice_list, r_min, r_max, is_rho_min=False,
                             img_path="%s/rho_max_t%02d.png" % (args.viz_dir, ti))

    # plot frs-probability
    if args.plot_frs_prob:
        prob_minimum = np.min(prob_min_list)
        prob_maximum = np.max(prob_max_list)
        prob_min_clip = 0.0
        p_min = min(prob_min_clip, prob_minimum)
        p_max = max(prob_min_clip, prob_maximum)
        plot_density_patches(prob_min_list, vertice_list, p_min, p_max, is_rho_min=True,
                             img_path="%s/prob_min_t%02d.png" % (args.viz_dir, ti))
        plot_density_patches(prob_max_list, vertice_list, p_min, p_max, is_rho_min=False,
                             img_path="%s/prob_max_t%02d.png" % (args.viz_dir, ti))

def backward_compute_fast(input_data):
    ti, i, Ao_query, bo_query, query_bbox, proj_in, proj_out, pack, args = input_data
    res={}
    res["ti"] = ti
    res["i"] = i
    Aout_t = pack["E"]
    bout_t = pack["f"]
    ineq = (np.concatenate((Aout_t, Ao_query), axis=0), np.concatenate((bout_t, bo_query), axis=0))
    if args.bbox_mode:
        depart_each = np.logical_or(pack["x_max"] < query_bbox[:, 0], query_bbox[:, 1] < pack["x_min"])
        intersect = not any(depart_each)
        res["bbox_intersect"] = intersect
        if intersect:
            vertices_out_full = sub_utils.get_A_b_vertices_robust(ineq[0], ineq[1], ti, i)

            # # TODO(debug)
            # print("CHECK")
            # print(pack["x_min"], pack["x_max"], pack["v_o"])
            # print("query box=",query_bbox)
            # print(vertices_out_full)
            res["v_o"], res["vol_o"], status = sub_utils.get_A_b_sys_vertices_volume_robust(vertices_out_full, is_input=False)
            vertices_out = project_polytope(proj_out, ineq, None)
            really_intersect = len(vertices_out) > 0
        else:
            really_intersect = False
            res["v_o"] = None
            res["vol_o"] = None
    else:
        vertices_out_full = sub_utils.get_A_b_vertices_robust(ineq[0], ineq[1], ti, i)
        res["v_o"], res["vol_o"], status = sub_utils.get_A_b_sys_vertices_volume_robust(vertices_out_full,
                                                                                        is_input=False)
        vertices_out = project_polytope(proj_out, ineq, None)
        really_intersect = len(vertices_out) > 0

    if really_intersect:
        _, rhomin, rhomax = sub_utils.non_empty(ineq[0], ineq[1])
        res["g_min"] = rhomin * ti * args.dt
        res["g_max"] = rhomax * ti * args.dt

        # output->input mapping
        ineq_new = (np.dot(ineq[0], pack["C"]), ineq[1] - np.dot(ineq[0], pack["d"]))
        if proj_in is not None:
            vertices_in = project_polytope(proj_in, ineq_new, None)
            res["v_i_2"] = vertices_in
        else:
            res["v_i_2"] = None
        res["nonzeros"] = True
    else:
        res["v_i_2"] = None
        res["nonzeros"] = False
    return res

def get_in_out_projs(ndim):
    E_in = np.zeros((2, ndim))
    E_out = np.zeros((2, ndim))
    f = np.zeros(2)
    E_in[0, 0] = 1.
    E_in[1, 1] = 1.
    E_out[0, 1] = 1.
    E_out[1, 2] = 1.

    proj_in = (E_in, f)  # proj(x) = E * x + f
    proj_out = (E_out, f)
    return proj_in, proj_out

def get_out_projs(ndim):
    E_out = np.zeros((ndim-1, ndim))
    for i in range(ndim-1):
        E_out[i, i+1]=1.
    f = np.zeros(ndim-1)
    return (E_out, f)

def check_brs_func(bag):
    ndim = len(args.q_mins)
    Ao_query=np.zeros((2*ndim, ndim))
    for i in range(ndim):
        Ao_query[i*2, i] = -1.0
        Ao_query[i*2+1, i] = 1.0
    bo_query_bak=np.zeros((2*ndim, ))
    for i in range(ndim):
        bo_query_bak[i*2] = -args.q_mins[i]
        bo_query_bak[i*2+1] = args.q_maxs[i]
    pool = Pool(processes=args.num_workers)

    # TODO (low/mid/high cases)

    nonzeros_states = 0
    num_bbox_intersect = 0
    total_prob_mins = []
    total_prob_maxs = []
    total_prob_mins_list = {}
    total_prob_maxs_list = {}
    total_vertices_list = {}

    proj_in, proj_out = get_in_out_projs(ndim)
    outputs_d={}

    check_time=0

    for ti in range(1, args.nt):
        bo_query = np.array(bo_query_bak)
        bo_query[0] /= (ti*args.dt)
        bo_query[1] /= (ti*args.dt)

        vertices_out_full = sub_utils.get_A_b_vertices_robust(Ao_query, bo_query, ti, i=-1)
        assert len(vertices_out_full) > 0
        vertices_array = np.stack(vertices_out_full, axis=0)
        upper = np.max(vertices_array, axis=0)
        lower = np.min(vertices_array, axis=0)
        query_bbox = np.stack((lower, upper), axis=-1)[1:]

        tt1=time.time()
        outputs = pool.map(backward_compute_fast, [(ti, i, Ao_query, bo_query, query_bbox, proj_in, proj_out, bag[ti][i], args) for i in range(len(bag[ti]))])
        tt2=time.time()
        check_time += (tt2-tt1)
        nonzeros_states += np.sum([od["nonzeros"] for od in outputs])
        if args.bbox_mode:
            num_bbox_intersect += np.sum([od["bbox_intersect"] for od in outputs])
        outputs_d[ti] = outputs

        dens_min_list = []
        dens_max_list = []
        volume_list = []
        prob_min_list = []
        prob_max_list = []
        state_i_list = []

        for od in outputs:
            if od["vol_o"] is not None:
                state_i_list.append(od["i"])
                dens_max_list.append(np.exp(od["g_max"]) * bag[ti][od["i"]]["init_dens"])
                dens_min_list.append(np.exp(od["g_min"]) * bag[ti][od["i"]]["init_dens"])
                volume_list.append(od["vol_o"])

        dens_max_list = np.array(dens_max_list)
        dens_min_list = np.array(dens_min_list)
        volume_list = np.array(volume_list)
        total_prob_mins_list[ti] = {}
        total_prob_maxs_list[ti] = {}
        total_vertices_list[ti] = {}
        for i in range(volume_list.shape[0]):
            # TODO(could have different solutions here)!
            # prob_min = bag[ti][state_i_list[i]]["prob_min"] / bag[ti][state_i_list[i]]["volume"] * \
            #            volume_list[i]
            # prob_max = bag[ti][state_i_list[i]]["prob_max"] / bag[ti][state_i_list[i]]["volume"] * \
            #            volume_list[i]

            # prob_min = bag[ti][state_i_list[i]]["rho_min"] * volume_list[i]
            # prob_max = bag[ti][state_i_list[i]]["rho_max"] * volume_list[i]

            prob_min = dens_min_list[i] * volume_list[i]
            prob_max = dens_max_list[i] * volume_list[i]

            prob_min_list.append(prob_min)
            prob_max_list.append(prob_max)
            total_prob_mins_list[ti][state_i_list[i]] = prob_min
            total_prob_maxs_list[ti][state_i_list[i]] = prob_max
            total_vertices_list[ti][state_i_list[i]] = outputs_d[ti][state_i_list[i]]["v_i_2"]
        total_prob_mins.append(np.sum(prob_min_list))
        total_prob_maxs.append(np.sum(prob_max_list))
        print("t=%02d wall:%.4f nonzeros:%d  intersect:%d" % (ti, check_time,nonzeros_states,num_bbox_intersect))

    print("total_pmin:", total_prob_mins)
    print("total_pmax:", total_prob_maxs)
    print("mean total_pmin:", np.mean(total_prob_mins))
    print("mean total_pmax:", np.mean(total_prob_maxs))

    # TODO
    for ti in range(1, args.nt):
        prob_min_clip = 0.0
        viz_queue = [(total_vertices_list[ti][state_i],
                      total_prob_mins_list[ti][state_i], total_prob_maxs_list[ti][state_i]) for state_i in
                     total_vertices_list[ti]]
        if len(viz_queue) == 0:
            prob_minimum = 1
            prob_maximum = 0
        else:
            prob_minimum = np.min([x[1] for x in viz_queue])
            prob_maximum = np.max([x[2] for x in viz_queue])
            prob_min = min(prob_min_clip, prob_minimum)
            prob_max = max(prob_min_clip, prob_maximum)

        total_prob_mins_ = [total_prob_mins_list[ti][state_i] for state_i in total_vertices_list[ti]]
        total_prob_maxs_ = [total_prob_maxs_list[ti][state_i] for state_i in total_vertices_list[ti]]
        total_vertices_ = [total_vertices_list[ti][state_i] for state_i in total_vertices_list[ti]]

        plot_density_patches(total_prob_mins_, total_vertices_, prob_min, prob_max, is_rho_min=True,
                             img_path="%s/bwd_pmin_t%02d.png" % (args.viz_dir, ti))
        plot_density_patches(total_prob_maxs_, total_vertices_, prob_min, prob_max, is_rho_min=False,
                             img_path="%s/bwd_pmax_t%02d.png" % (args.viz_dir, ti))
    pool.close()

def get_xmin_xmax_ymin_ymax(args):
    if args.exp_mode=="robot":
        xmin=-2.0
        xmax=2.0
        ymin=-2.0
        ymax=2.0
    if args.exp_mode=="vdp":
        xmin=-5
        xmax=5
        ymin=-5
        ymax=5
    if args.exp_mode=="dint":
        xmin=-8
        xmax=8
        ymin=-8
        ymax=8
    if args.exp_mode=="acc":
        xmin=30.0
        xmax=80.0
        ymin=0.0
        ymax=30.0
    if args.exp_mode=="car":
        xmin=-8.0
        xmax=8.0
        ymin=-8.0
        ymax=8.0

    ratio=1.25
    xmin = xmin * ratio
    xmax = xmax * ratio
    ymin = ymin * ratio
    ymax = ymax * ratio

    return xmin, xmax, ymin, ymax


def plot_varied_frs_func(data):
    x_min,x_max,y_min,y_max=get_xmin_xmax_ymin_ymax(args)
    args.x_min = x_min
    args.x_max = x_max
    args.y_min = y_min
    args.y_max = y_max

    cmap = matplotlib.cm.inferno
    volume_t_alpha_list=[]

    rpm_prob_list_list=[]
    rpm_save_list_list=[]
    rpm_volume_list_list=[]
    rpm_indices_list_list=[]
    rpm_check_thres_list_list=[]
    r_rall_list=[]
    eps_list=[]
    thres_list=[]

    for t_iii, ti in enumerate(args.viz_timesteps):
        rho_min_list= data[ti]["dens_min_list"]
        rho_max_list = data[ti]["dens_max_list"]
        prob_min_list = data[ti]["prob_min_list"]
        prob_max_list = data[ti]["prob_max_list"]
        vertice_list = data[ti]["vertice_list"]
        volume_list = data[ti]["volume_list"]
        volume_t_alpha_list.append([])
        r_rall_list.append([])
        eps_list.append([])
        thres_list.append([])
        rho_min_clip = 0.0
        prob_minimum = np.min(prob_min_list)
        prob_maximum = np.max(prob_max_list)
        prob_min = min(rho_min_clip, prob_minimum)
        prob_max = max(rho_min_clip, prob_maximum)
        if args.advance_plot:
            desired_levels = [1.0, 0.99, 0.995, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1]

            if args.select_by_dens:
                indices = np.argsort(rho_max_list)
            else:
                indices = np.argsort(prob_max_list)

            prev_prob = 1.0
            total_volume = np.sum([volume_list[iii] for iii in range(len(prob_max_list))])
            standard_volume = total_volume

            rpm_prob_list_list.append([prev_prob])
            rpm_save_list_list.append([0.0])
            rpm_volume_list_list.append([standard_volume])
            rpm_indices_list_list.append([indices])
            rpm_check_thres_list_list.append([0.0])

            exclude_indices=[]

            for ind in indices:
                exclude_indices.append(ind)
                include_indices = [iii for iii in indices]
                for del_i in exclude_indices:
                    include_indices.remove(del_i)
                prob = np.sum([prob_min_list[iii] for iii in include_indices]) / (
                        np.sum([prob_min_list[iii] for iii in include_indices]) + np.sum(
                    [prob_max_list[iii] for iii in exclude_indices])
                )

                exclude_volume = np.sum([volume_list[iii] for iii in exclude_indices])
                ratio = exclude_volume / total_volume

                for level in desired_levels:
                    if prev_prob >= level and prob < level:
                        rpm_volume_list_list[-1].append(total_volume * (1 - ratio))
                        rpm_prob_list_list[-1].append(prob)
                        rpm_save_list_list[-1].append(ratio)
                        rpm_indices_list_list[-1].append(include_indices)
                        if args.select_by_dens:
                            rpm_check_thres_list_list[-1].append(rho_max_list[ind])
                        else:
                            rpm_check_thres_list_list[-1].append(prob_max_list[ind])

                        print("t=%02d  ignored=%d  prob=%.4f  save=%.4f currV:%.9f originalV:%.9f" % (
                            ti, len(exclude_indices), prob, ratio, total_volume - exclude_volume, total_volume))
                        break


            # viz out
            for draw in ["prob", "density"]:
                i = t_iii
                for l_i, level_set in enumerate(desired_levels):
                    for p_i, prob in enumerate(rpm_prob_list_list[i]):
                        if p_i > 0 and prob < level_set and rpm_prob_list_list[i][p_i - 1] >= level_set:
                            draw_background()
                            r_rall_list[-1].append(rpm_volume_list_list[i][p_i - 1] / standard_volume)
                            eps_list[-1].append(rpm_prob_list_list[i][p_i - 1])
                            thres_list[-1].append(rpm_check_thres_list_list[i][l_i])

                            if draw=="prob":
                                prob_min = np.min([data[ti]["prob_min_list"][vert_idx] for vert_idx in rpm_indices_list_list[i][p_i - 1]])
                                prob_max = np.max([data[ti]["prob_max_list"][vert_idx] for vert_idx in rpm_indices_list_list[i][p_i - 1]])
                                sort_indices = np.argsort(data[ti]["prob_max_list"])
                            else:
                                rho_min = np.min([data[ti]["dens_min_list"][vert_idx] for vert_idx in rpm_indices_list_list[i][p_i - 1]])
                                rho_max = np.max([data[ti]["dens_max_list"][vert_idx] for vert_idx in rpm_indices_list_list[i][p_i - 1]])
                                sort_indices = np.argsort(data[ti]["dens_max_list"])

                            tmp_vertes = []
                            for vert_idx in sort_indices:
                                vert = data[ti]["vertice_list"][vert_idx]
                                if vert_idx in rpm_indices_list_list[i][p_i - 1]:
                                    if draw == "prob":
                                        norm_rho = (data[ti]["prob_max_list"][vert_idx] - prob_min) / (
                                            np.clip(prob_max - prob_min, a_min=1e-4, a_max=1e10))
                                    else:
                                        if rho_max == rho_min:
                                            norm_rho = (data[ti]["dens_max_list"][vert_idx] - 0) / (
                                                np.clip(rho_max - 0, a_min=1e-4, a_max=1e10))
                                        else:
                                            norm_rho = (data[ti]["dens_max_list"][vert_idx] - rho_min) / (
                                                np.clip(rho_max - rho_min, a_min=1e-4, a_max=1e10))
                                    color_i = cmap(norm_rho)
                                    plot_polygon(vert, color=color_i, alpha=1.0)
                                else:
                                    tmp_vertes.append(vert)

                            for vert in tmp_vertes:
                                plot_polygon(vert, alpha=1.0, color="limegreen", fill=False)

                            break

                    ax = plt.gca()
                    ax.set_aspect(1.0 / ((args.y_max - args.y_min) / (args.x_max - args.x_min)), adjustable='box')
                    plt.xlim(args.x_min, args.x_max)
                    plt.ylim(args.y_min, args.y_max)

                    if draw == "prob":
                        norm = matplotlib.colors.Normalize(vmin=prob_min, vmax=prob_max)
                    else:
                        norm = matplotlib.colors.Normalize(vmin=rho_min, vmax=rho_max)

                    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
                    plt.title("Ours vol:%.2fX (p=%.2f.>%.2f)" % (
                    rpm_volume_list_list[i][p_i - 1] / standard_volume, rpm_prob_list_list[i][p_i - 1], level_set))
                    plt.tight_layout()

                    cat_str = "heat_with"

                    if draw == "prob":
                        plt.savefig(os.path.join(args.viz_dir,
                                                 "%s_4_rpm_prob%d_%s_%03d.png" % (args.exp_mode, l_i + 1, cat_str, ti)),
                                    bbox_inches='tight', pad_inches=0)
                    else:
                        plt.savefig(
                            os.path.join(args.viz_dir, "%s_4_rpm_rho%d_%s_%03d.png" % (args.exp_mode, l_i + 1, cat_str, ti)),
                            bbox_inches='tight', pad_inches=0)
                    plt.close()

        elif args.plot_mesh:

            font = {
                # 'family' : 'normal',
                # 'weight' : 'bold',
                'size': 16
            }
            matplotlib.rc('font', **font)

            for a_i, alpha in enumerate(args.alphas):
                draw_background()
                exclude_indices = []
                if args.select_by_dens:
                    indices = np.argsort(rho_max_list)
                else:
                    indices = np.argsort(prob_max_list)
                cnt = 0
                include_indices=[]
                for i in indices:
                    if (args.select_by_dens and rho_max_list[i] < alpha) or (args.select_by_dens==False and prob_max_list[i] < alpha):
                        exclude_indices.append(i)
                    else:
                        vertices = vertice_list[i]
                        include_indices.append(i)
                        # # print(vertices)
                        # norm_rho = (prob_max_list[i] - prob_min) / (np.clip(prob_max - prob_min, a_min=1e-4, a_max=1e10))
                        # color_i = cmap(norm_rho)
                        # color_i = (color_i[0], color_i[1], color_i[2], 1.0)
                        # plot_polygon(vertices, color=color_i, alpha=color_i[-1])
                        # # plot_polygon(vertices, color=(1.0, 1.0, 0.0), alpha=color_i[-1])
                    cnt += 1

                # include_indices = list(set(indices) - set(exclude_indices))
                rho_min = np.min(rho_min_list)
                rho_max = np.max(rho_max_list)

                for i in np.argsort(rho_max_list):
                    if i in include_indices:
                        vertices = vertice_list[i]
                        if args.plot_mesh_log:
                            if args.plot_mesh_prob:
                                norm_rho = (np.log(prob_max_list[i]) - np.log(prob_min)) / (
                                    np.clip(np.log(prob_max) - np.log(prob_min), a_min=1e-4, a_max=1e10))
                            else:
                                norm_rho = (np.log(rho_max_list[i]) - np.log(rho_min)) / (
                                    np.clip(np.log(rho_max) - np.log(rho_min), a_min=1e-4, a_max=1e10))
                        else:
                            if args.plot_mesh_prob:
                                norm_rho = (prob_max_list[i] - prob_min) / (np.clip(prob_max - prob_min, a_min=1e-4, a_max=1e10))
                            else:
                                norm_rho = (rho_max_list[i] - rho_min) / (np.clip(rho_max - rho_min, a_min=1e-4, a_max=1e10))
                        color_i = cmap(norm_rho)
                        color_i = (color_i[0], color_i[1], color_i[2], 1.0)
                        plot_polygon(vertices, color=color_i, alpha=color_i[-1])
                        # plot_polygon(vertices, color=(1.0, 1.0, 0.0), alpha=color_i[-1])

                    if i in exclude_indices:
                        vertices = vertice_list[i]
                        plot_polygon(vertices, alpha=1.0, color="limegreen", fill=False)


                prob = np.sum([prob_min_list[i] for i in include_indices]) / (
                        np.sum([prob_min_list[i] for i in include_indices]) + np.sum(
                    [prob_max_list[i] for i in exclude_indices])
                )

                total_volume = np.sum([volume_list[i] for i in range(len(prob_max_list))])
                exclude_volume = np.sum([volume_list[i] for i in exclude_indices])
                ratio = exclude_volume / total_volume

                print("t=%02d  alpha=%.13f  ignored=%d  prob=%.4f  save=%.4f currV:%.9f originalV:%.9f" % (ti, alpha, len(exclude_indices), prob, ratio, total_volume-exclude_volume, total_volume))

                plt.xlim(args.x_min, args.x_max)
                plt.ylim(args.y_min, args.y_max)
                ax = plt.gca()
                ax.set_aspect(1.0 / ((args.y_max-args.y_min)/(args.x_max-args.x_min)), adjustable='box')
                if args.plot_mesh_log:
                    if args.plot_mesh_prob:
                        norm = matplotlib.colors.Normalize(vmin=np.log(prob_min), vmax=np.log(prob_max))
                    else:
                        norm = matplotlib.colors.Normalize(vmin=np.log(rho_min), vmax=np.log(rho_max))
                else:
                    if args.plot_mesh_prob:
                        norm = matplotlib.colors.Normalize(vmin=prob_min, vmax=prob_max)
                    else:
                        norm = matplotlib.colors.Normalize(vmin=rho_min, vmax=rho_max)

                plt.xlabel("x")
                plt.ylabel("y")
                plt.title("Ours vol:=%.2fX (p>%.4f)"%(1-ratio, prob))
                plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
                plt.savefig(ospj(args.viz_dir, "prob_frs%03d_a%.13f_prob%.6f_save%.6f.png" % (ti, alpha, prob, ratio)),
                            bbox_inches='tight', pad_inches=0)
                plt.close()

                volume_t_alpha_list[-1].append(total_volume-exclude_volume)

        else:
            for a_i, alpha in enumerate(args.alphas):

                draw_background()
                exclude_indices = []
                if args.select_by_dens:
                    indices = np.argsort(rho_max_list)
                else:
                    indices = np.argsort(prob_max_list)
                cnt = 0
                for i in indices:
                    if (args.select_by_dens and rho_max_list[i] < alpha) or (args.select_by_dens==False and prob_max_list[i] < alpha):
                        exclude_indices.append(i)
                    else:
                        vertices = vertice_list[i]
                        # print(vertices)
                        norm_rho = (prob_max_list[i] - prob_min) / (np.clip(prob_max - prob_min, a_min=1e-4, a_max=1e10))
                        color_i = cmap(norm_rho)
                        color_i = (color_i[0], color_i[1], color_i[2], 1.0)
                        # plot_polygon(vertices, color=color_i, alpha=color_i[-1])
                        plot_polygon(vertices, color=(1.0, 1.0, 0.0), alpha=color_i[-1])
                    cnt += 1

                include_indices = list(set(indices) - set(exclude_indices))

                prob = np.sum([prob_min_list[i] for i in include_indices]) / (
                        np.sum([prob_min_list[i] for i in include_indices]) + np.sum(
                    [prob_max_list[i] for i in exclude_indices])
                )

                total_volume = np.sum([volume_list[i] for i in range(len(prob_max_list))])
                exclude_volume = np.sum([volume_list[i] for i in exclude_indices])
                ratio = exclude_volume / total_volume

                print("t=%02d  alpha=%.13f  ignored=%d  prob=%.4f  save=%.4f currV:%.9f originalV:%.9f" % (ti, alpha, len(exclude_indices), prob, ratio, total_volume-exclude_volume, total_volume))

                plt.xlim(args.x_min, args.x_max)
                plt.ylim(args.y_min, args.y_max)
                ax = plt.gca()
                ax.set_aspect(1.0 / ((args.y_max-args.y_min)/(args.x_max-args.x_min)), adjustable='box')
                norm = matplotlib.colors.Normalize(vmin=prob_min, vmax=prob_max)
                plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
                plt.savefig(ospj(args.viz_dir, "prob_frs%03d_a%.13f_prob%.6f_save%.6f.png" % (ti, alpha, prob, ratio)))
                plt.close()

                volume_t_alpha_list[-1].append(total_volume-exclude_volume)


def volume_hyper_ellipsoid(radius, scales):
    ndim = scales.shape[0]
    if ndim==2:
        v_sphere = np.pi * (radius ** 2)
    else:
        v_sphere = 2.0/ndim * (np.pi ** (ndim/2.0)) * (radius ** ndim) / scipy.special.gamma(ndim/2.0)

    v_ellipsoid = v_sphere * np.product(scales)

    return v_ellipsoid

def volume_hyper_rect(radius, scales):
    ndim = scales.shape[0]
    return (radius ** ndim) * np.product(scales)

def uniform_sample_ball(N, mean, scale):
    ndim = mean.shape[0]
    u = np.random.normal(0, 1, (N, ndim))  # an array of d normally distributed random variables
    norm = np.linalg.norm(u, axis=-1).reshape((-1, 1))
    r = np.random.rand(N, 1) ** (1.0/ndim)
    x = r * u / norm
    return x * scale + mean

def uniform_sample_rect(N, mean, scale):
    ndim = mean.shape[0]
    x = np.random.rand(N, ndim)  # an array of d normally distributed random variables
    return x * 2 * scale - scale + mean


def minmax_2_A_b(q_mins, q_maxs):
    ndim = len(q_mins)
    Ao_query = np.zeros((2 * ndim, ndim))
    for i in range(ndim):
        Ao_query[i * 2, i] = -1.0
        Ao_query[i * 2 + 1, i] = 1.0
    bo_query_bak = np.zeros((2 * ndim,))
    for i in range(ndim):
        bo_query_bak[i * 2] = -q_mins[i]
        bo_query_bak[i * 2 + 1] = q_maxs[i]
    return Ao_query, bo_query_bak

class MockArgs:
    pass

def density_est_func(input_data):
    ti, x_min, x_max, y_min, y_max, in_max, in_min, out_means, out_stds, n_sim_trajs, sim_s_list, rho_list, bag_ti, kde_str, init_s_list = input_data
    res={}
    # print(ti)

    # setup
    res["ti"] = ti
    ref = sim_s_list[args.track_id, :]
    if args.track_type=="ball":
        sim_in = np.where(np.linalg.norm(sim_s_list[:, :] - ref, axis=-1) < args.track_radius)[0]
        original_volume = volume_hyper_ellipsoid(args.track_radius, out_stds[1:])
        testing_points = uniform_sample_ball(args.n_sampling_points, ref, args.track_radius)
    elif args.track_type=="rect":
        sim_in = np.where(np.linalg.norm(sim_s_list[:, :] - ref, axis=-1, ord=np.inf) < args.track_radius)[0]
        original_volume = volume_hyper_rect(args.track_radius, out_stds[1:])
        print("DEBUG original volume from %.4f radius is: %.4f"%(args.track_radius, original_volume))
        testing_points = uniform_sample_rect(args.n_sampling_points, ref, args.track_radius)
    normalize_term = np.product(in_max[:-1] - in_min[:-1])
    print("DEBUG normalized term is %.4f" % normalize_term)

    # simulation-based density
    sim_prob = 1.0 * sim_in.shape[0] / n_sim_trajs
    res["sim_prob"] = sim_prob
    print("DEBUG sim_prob is %.4f" % sim_prob)

    # liouville-based density
    assert args.distribution == "uniform"
    # random_idx = np.random.choice(sim_in, args.n_sampling_points)
    # print("DEBUG ti",ti,"random", np.min(rho_list[random_idx]), np.max(rho_list[random_idx]), np.mean(rho_list[random_idx]), np.std(rho_list[random_idx]))
    # liou_avg_dens = np.mean(rho_list[random_idx])
    # liou_prob = liou_avg_dens * original_volume / normalize_term
    # liou_prob = np.clip(liou_prob, 0.0, 1.0)
    # res["liou_prob"] = liou_prob

    # kde-based density
    # TODO(to give more trials, options for KDE)
    for k_config in kde_str:
        # TODO
        kernel='epanechnikov'
        bandwidth=float(k_config.split("_")[-1])

        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(sim_s_list[:, :])
        # kde = KernelDensity(kernel='epanechnikov', bandwidth=0.01).fit(sim_s_list[:, :])
        kde_avg_dens = np.mean(np.exp(kde.score_samples(testing_points)))
        kde_prob = kde_avg_dens * original_volume / normalize_term
        kde_prob = np.clip(kde_prob, 0.0, 1.0)
        res[k_config] = kde_prob


    # nn-rpm based density  # TODO first dim rho_min, rho_max
    lower = np.concatenate((np.array([-10.0]), (ref - args.track_radius)*out_stds[1:]+out_means[1:]))
    upper = np.concatenate((np.array([100.0]), (ref + args.track_radius)*out_stds[1:]+out_means[1:]))
    Aq, bq = minmax_2_A_b(lower, upper)

    # #TODO(debug)
    # if ti==0:
    #     print("CHECK origin ref=",ref *out_stds[1:]+out_means[1:])
    #     exit()

    if ti>0:
        bq[0] /= (ti * args.dt)
        bq[1] /= (ti * args.dt)
        lower[0] /= (ti * args.dt)
        upper[0] /= (ti * args.dt)

    query_bbox = np.stack((lower, upper), axis=-1)
    tt1 = time.time()
    pool = Pool(processes=args.num_workers)
    outputs = pool.map(backward_compute_fast,
                       [(ti, i, Aq, bq, query_bbox, None, get_out_projs(out_stds.shape[0]), bag_ti[i], args) for i in
                        range(len(bag_ti))])
    tt2 = time.time()

    # import torch
    # from train_nn import Net
    # mock_args=MockArgs()
    # mock_args.input_dim=3
    # mock_args.hiddens=[32,32]
    # mock_args.normalize=False
    # mock_args.t_struct=True
    # mock_args.only_density_arch=False
    # mock_args.only_dynamics_arch = False
    # net=Net(mock_args)

    # # dbg_points = np.random.rand(1000, 2)
    # # dbg_points[:, 0] = dbg_points[:, 0] * 0.5 + 0.5
    # # dbg_points[:, 1] = dbg_points[:, 1] * 0.5 + 0.5
    # dbg_points = init_s_list[sim_in,:]
    # dbg_ts = np.ones_like(dbg_points[:,0:1]) * ti * args.dt
    # x_tensor = torch.from_numpy(np.concatenate((dbg_points, dbg_ts), axis=-1)).float()
    # o_tensor = net(x_tensor)
    # o_pts = o_tensor.detach().cpu().numpy()[:,1:]

    # color_list=np.random.rand(1000,3)
    # for ii,pack in enumerate(bag_ti): #enumerate(outputs):
    #     if pack["v_o"] is not None:
    #         plot_polygon(pack["v_i"], color=color_list[ii])
    # plt.scatter(sim_s_list[sim_in,0], sim_s_list[sim_in,1],s=1.0,c='blue', alpha=0.1)
    # # plt.scatter(o_pts[:, 0], o_pts[:, 1], s=1.0, c='red', alpha=0.5)
    # plt.xlim(-0.0, 2.0)
    # plt.ylim(-0.0, 2.0)
    # plt.axis("scaled")
    # plt.show()

    nonzeros_states = np.sum([od["nonzeros"] for od in outputs])
    if args.bbox_mode:
        num_bbox_intersect = np.sum([od["bbox_intersect"] for od in outputs])
    dens_min_list = []
    dens_max_list = []
    volume_list = []
    prob_min_list = []
    prob_max_list = []
    state_i_list = []

    for od in outputs:
        if od["vol_o"] is not None:
            state_i_list.append(od["i"])
            dens_max_list.append(np.exp(od["g_max"]) * bag_ti[od["i"]]["init_dens"])
            dens_min_list.append(np.exp(od["g_min"]) * bag_ti[od["i"]]["init_dens"])
            volume_list.append(od["vol_o"])

    dens_max_list = np.array(dens_max_list)
    dens_min_list = np.array(dens_min_list)
    volume_list = np.array(volume_list)
    for i in range(volume_list.shape[0]):
        prob_min = dens_min_list[i] * volume_list[i]
        prob_max = dens_max_list[i] * volume_list[i]
        prob_min_list.append(prob_min)
        prob_max_list.append(prob_max)

    res["nn_min"] = np.clip(np.sum(prob_min_list), 0, 1)
    res["nn_max"] = np.clip(np.sum(prob_max_list), 0, 1)
    # print("RPM ti=%02d wall:%.4f nonzeros:%d  intersect:%d sim:%.6f kde:%.6f, pmin:%.6f pmax:%.6f"
    #       % (ti, tt2-tt1, nonzeros_states, num_bbox_intersect, res["sim_prob"], res[kde_str[0]], res["nn_min"], res["nn_max"]))


    # scatter
    plt.scatter(sim_s_list[:, args.x_index], sim_s_list[:, args.y_index], color="b", alpha=0.5, s=1.0)
    plt.scatter(sim_s_list[sim_in, args.x_index], sim_s_list[sim_in, args.y_index], color="r", alpha=0.5, s=1.0)
    ax = plt.gca()
    ax.set_aspect(1.0 / ((y_max - y_min) / (x_max - x_min)), adjustable='box')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel(args.x_label)
    plt.ylabel(args.y_label)
    plt.savefig(os.path.join(args.viz_dir, "tmp%03d.png" % (ti)), bbox_inches='tight', pad_inches=0)
    plt.close()
    pool.close()
    return res

def run_density_measure_exp_func(bag, saved_data):
    # TODO (num_samples, nt, ndim)
    sim_data = np.load(args.sim_data_path, allow_pickle=True)['arr_0'].item()
    n_sim_trajs = sim_data['s_list'].shape[0]

    train_num_traj = int(n_sim_trajs * 0.8)
    in_means, in_stds, out_means, out_stds, in_min, in_max, out_min, out_max = \
        sub_utils.get_data_stat(sim_data, train_num_traj, include_minmax=True)

    if args.normalize:
        sim_s_list = (sim_data['s_list'] - out_means[1:]) / out_stds[1:]
        x_min = (args.x_min - out_means[1 + args.x_index]) / out_stds[1 + args.x_index]
        x_max = (args.x_max - out_means[1 + args.x_index]) / out_stds[1 + args.x_index]
        y_min = (args.y_min - out_means[1 + args.y_index]) / out_stds[1 + args.y_index]
        y_max = (args.y_max - out_means[1 + args.y_index]) / out_stds[1 + args.y_index]
    else:
        sim_s_list = sim_data['s_list']
        in_means = in_means * 0.0
        out_means = out_means * 0.0
        out_stds = out_stds * 0.0 + 1
        in_stds = in_stds * 0.0 + 1
        x_min, x_max, y_min, y_max = args.x_min, args.x_max, args.y_min, args.y_max

    #t_span=[0,15,30,45]  #list(range(0, args.nt, 5))
    t_span =list(range(args.debug_start_from, args.nt, args.debug_gap))
    # t_span = [0, 0]
    # t_span = list(range(0, args.nt, 1))

    ti2i = {ti:i for i,ti in enumerate(t_span)}

    kde_configs=[("epanechnikov", 0.001), ("epanechnikov", 0.01), ("epanechnikov", 0.1)]
    # kde_configs = [("epanechnikov", 0.01), ("epanechnikov", 0.1), ("epanechnikov", 0.5), ("epanechnikov", 1.0), ("epanechnikov", 2.0)]
    kde_str = []
    meas_list={}
    meas_list["sim_prob"] = [0.0] * len(t_span)
    for ker, bw in kde_configs:
        kde_str.append("kde_%s_%.4f"%(ker[:3], bw))
        meas_list[kde_str[-1]] = [0.0] * len(t_span)
    meas_list["nn_min"] = [0.0] * len(t_span)
    meas_list["nn_max"] = [0.0] * len(t_span)

    # sim_prob_list = [0.0] * len(t_span)
    # liou_prob_list = [0.0] * len(t_span)
    # kde_prob_list = [0.0] * len(t_span)
    # nn_prob_min_list = [0.0] * len(t_span)
    # nn_prob_max_list = [0.0] * len(t_span)

    input_data = [(ti, x_min, x_max, y_min, y_max, in_max, in_min, out_means, out_stds, n_sim_trajs, sim_s_list[:,ti], sim_data["rho_list"][:,ti], bag[ti], kde_str, sim_s_list[:,0]) for ti in t_span]
    # pool = Pool(processes=args.num_workers)
    # res_list = pool.map(density_est_func, input_data)
    # pool.close()
    res_list=[]
    for input_res in input_data:
        res_list.append(density_est_func(input_res))

    for res in res_list:
        ti = res["ti"]
        for key in meas_list.keys():
            meas_list[key][ti2i[ti]] = res[key]
        print("ti:%02d sim:%.6f kde:%s min:%.6f max:%.6f"%(ti, res["sim_prob"],
                                                           "|".join(["%.6f"%(res[kk]) for kk in kde_str]),
                                                           res["nn_min"], res["nn_max"]))

    # for res in res_list:
    #     ti = res["ti"]
    #     sim_prob_list[ti2i[ti]] = res["sim_prob"]
    #     # liou_prob_list[ti2i[ti]] = res["liou_prob"]
    #     kde_prob_list[ti2i[ti]] = res["kde_prob"]
    #     nn_prob_min_list[ti2i[ti]] = res["nn_prob_min"]
    #     nn_prob_max_list[ti2i[ti]] = res["nn_prob_max"]
    #     print("ti:%02d sim:%.9f kde:%.9f min:%.9f max:%.9f"%(ti, res["sim_prob"], res["kde_prob"],  res["nn_prob_min"], res["nn_prob_max"]))

    color_list=["indianred", "brown", "darkred", "tomato", "salmon"]

    for k in range(2, len(t_span)):
        # plot curve
        plt.plot(t_span[:k], meas_list["sim_prob"][:k], label="sim", color="blue")
        # plt.plot(t_span, liou_prob_list[:k], label="liou", color="springgreen")
        for ki, kde_s in enumerate(kde_str):
            plt.plot(t_span[:k], meas_list[kde_s][:k], label=kde_s, color=color_list[ki])
        plt.plot(t_span[:k], meas_list["nn_min"][:k], label="ours_min", color="limegreen")
        plt.plot(t_span[:k], meas_list["nn_max"][:k], label="ours_max", color="darkgreen")
        plt.xlabel("ts")
        plt.ylabel("probability")
        plt.legend()
        plt.savefig(os.path.join(args.viz_dir, "prob_curve_%d.png"%k), bbox_inches='tight', pad_inches=0)
        plt.close()

    print_prob_metrics(meas_list["sim_prob"], meas_list["nn_min"], meas_list["nn_max"])

def print_prob_metrics(x_gt, x_min, x_max):
    x_gt = np.array(x_gt)
    x_min = np.array(x_min)
    x_max = np.array(x_max)

    success_rate = np.mean(np.logical_and(x_gt<x_max, x_gt>x_min))
    bound = np.mean(np.minimum(np.abs(x_gt - x_min), np.abs(x_gt-x_max)))
    rel_bound = np.mean(np.minimum(np.abs(x_gt - x_min), np.abs(x_gt-x_max)) / np.clip(x_gt, 0.1, 1.0))
    print("cover:%.4f bound:%.4f rel_bound:%.4f"%(success_rate, bound, rel_bound))

def smart_load_traj(path):
    if ".npz" not in path:
        the_path = os.path.join("..", 'data',path, 'traj_data.npz')
    elif len(path.split("/")) == 2:
        the_path = os.path.join("..", 'data', path)
    else:
        the_path = path
    return np.load(the_path, allow_pickle=True)['arr_0'].item()

from torch import nn
class LPNet(nn.Module):
    def __init__(self, args):
        super(LPNet, self).__init__()
        self.relu = nn.ReLU()
        self.args = args
        self.linear_list = nn.ModuleList()
        for hid_i, hid in enumerate(args.hiddens):
            if hid_i==0:
                self.linear_list.append(nn.Linear(args.input_dim, hid))
            else:
                self.linear_list.append(nn.Linear(args.hiddens[hid_i-1], hid))
        self.linear_list.append(nn.Linear(args.hiddens[-1], args.input_dim))
        # self.linear_list.append(nn.Linear(5, 32))
        # self.linear_list.append(nn.Linear(32, 32))
        # self.linear_list.append(nn.Linear(32, 5))

    def forward(self, x):
        for i in range(len(self.args.hiddens)):
            x = self.relu(self.linear_list[i](x))
        x = self.linear_list[-1](x)
        return x

def compute_bbox_volume(bbox0, bbox1):  # shape (ndim, 2)
    return np.product(bbox1 - bbox0)


def bbox_2_vert(bbox, index1=0, index2=1):  # (N, 2)
    x_min = bbox[index1, 0]
    x_max = bbox[index1, 1]
    y_min = bbox[index2, 0]
    y_max = bbox[index2, 1]
    return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]


def proj_xy(vert, x_index=0, y_index=1):
    return (vert[x_index], vert[y_index])

def run_reach_func(bags, saved_data):
    cvx_volume_list = []
    cvx_vertices_list = []
    rlp_volume_list = []
    rlp_upper_vol_list = []
    rlp_bbox_list = []
    final_data={}
    if args.plot_all:
        # assert "cvx" in args.method
        # assert "reachlp" in args.method
        # assert "rpm" in args.method
        x_min, x_max, y_min, y_max = get_xmin_xmax_ymin_ymax(args)
        args.x_min = x_min
        args.x_max = x_max
        args.y_min = y_min
        args.y_max = y_max
        cmap = matplotlib.cm.inferno
        final_data["args"] = args

    if "cvx" in args.method:
        final_data["cvx"]={}
        traj_data=smart_load_traj(args.traj_data_path)["s_list"]
        for ti in args.viz_timesteps:
            points = traj_data[:args.max_sample, ti]
            try:
                cvx_hull = ConvexHull(points=points)  # remove the indicator dim and the first dim (density gain)
                volume = cvx_hull.volume
            except scipy.spatial.qhull.QhullError:
                print("simplex V=0, just remove it")
                volume = 0.0
            cvx_volume_list.append(volume)
            cvx_vertices_list.append([traj_data[vert_i,ti,:] for vert_i in cvx_hull.vertices])
            print("ti:%03d vol:%.9f"%(ti, volume))
        final_data["cvx"]["cvx_volume_list"] = cvx_volume_list
        final_data["cvx"]["cvx_vertices_list"] = cvx_vertices_list
        np.savez(os.path.join(args.exp_dir_full,"cvx_volume_list"), cvx_volume_list)
        np.savez(os.path.join(args.exp_dir_full,"cvx_vertices_list"), cvx_vertices_list)

    if "reachlp" in args.method: # TODO reachLP
        import torch
        import nn_partition.analyzers as analyzers
        final_data["reachlp"]={}
        import socket
        host_name = socket.gethostname()

        old_args = np.load(os.path.join(os.path.dirname(args.pretrained_path),"..","args.npz"), allow_pickle=True)['args'].item()

        args.hiddens = old_args.hiddens
        dims_dict={
            "robot":5,
            "vdp":3,
            "dint":3,
            "car":5,
            "quad":7,
            "pend":5,
            "acc": 8,
            "toon":17,
            "gcas":14,
        }
        args.input_dim = dims_dict[args.exp_mode]

        torch_model = LPNet(args)

        m = torch.load(args.pretrained_path)
        torch_model.load_state_dict(m)
        torch_model.eval()
        input_range = np.stack((np.array(args.s_mins+[0.0]), np.array(args.s_maxs+[(args.nt-1)*args.dt])), axis=-1)

        partitioner_hyperparams = {
            "num_simulations": args.num_simulations,
            "type": args.partitioner,
            "termination_condition_type": args.term_type,
            "termination_condition_value": args.term_val,
            "interior_condition": args.interior_condition,
            "make_animation": False,
            "show_animation": False,
            "show_input": False,
            "show_output": False,
        }

        propagator_hyperparams = {
            "type": args.propagator,
            "input_shape": input_range.shape[:-1],
        }

        analyzer = analyzers.Analyzer(torch_model)
        analyzer.partitioner = partitioner_hyperparams
        analyzer.propagator = propagator_hyperparams


        for ti in args.viz_timesteps:
            t0=time.time()
            new_input_range = np.array(input_range)
            ndim = new_input_range.shape[1]-1
            new_input_range[-1, 0] = ti*args.dt
            new_input_range[-1, 1] = ti*args.dt
            output_range, analyzer_info = analyzer.get_output_range(new_input_range)
            out_list=[]
            XMIN=np.ones((ndim,)) * np.inf
            XMAX=np.ones((ndim,)) * (-np.inf)
            for info in analyzer_info["all_partitions"]:
                bbox_o = np.array(info[1])[1:, :]  # (ndim, 2)
                XMIN = np.minimum(XMIN, bbox_o[:, 0])
                XMAX = np.maximum(XMAX, bbox_o[:, 1])
                out_list.append(bbox_o)
            # print("xmin", list(XMIN))
            # print("xmax", list(XMAX))
            # just use all volumes as an upper bound
            upper_v = 0.0
            for bbox_o in out_list:
                upper_v += compute_bbox_volume(bbox_o[:, 0], bbox_o[:, 1])
            rlp_upper_vol_list.append(upper_v)
            # use monte carlo to get smaller region
            points = np.random.rand(args.n_mc, ndim)
            points = points * (XMAX-XMIN) + XMIN
            BIG_V = compute_bbox_volume(XMIN, XMAX)
            mask_list=[]
            for bbox_o in out_list:
                in_mask = np.logical_and(points<=bbox_o[:, 1], points>=bbox_o[:, 0])
                mask_list.append(np.product(in_mask, axis=-1))  # satisfy all dims
            mask_list = np.sum(np.stack(mask_list, axis=-1), axis=-1)>0 # satisfy any of the bbox
            volume = np.mean(mask_list) * BIG_V
            rlp_volume_list.append(volume)
            rlp_bbox_list.append(out_list)
            print("ti:%03d mc-vol:%.9f up-vol:%.9f whole:%.9f" % (ti, volume, upper_v, BIG_V))

        final_data["reachlp"]["rlp_volume_list"]=rlp_volume_list
        final_data["reachlp"]["rlp_bbox_list"] = rlp_bbox_list
        np.savez(os.path.join(args.exp_dir_full, "rlp_volume_list"), rlp_volume_list)
        np.savez(os.path.join(args.exp_dir_full, "rlp_upper_vol_list"), rlp_upper_vol_list)
        np.savez(os.path.join(args.exp_dir_full, "rlp_bbox_list"), rlp_bbox_list)

    if "rpm" in args.method:
        final_data["rpm"]={}
        volume_t_alpha_list = []
        data=saved_data
        desired_levels=[1.0, 0.99, 0.995, 0.9, 0.85, 0.8, 0.6, 0.5, 0.3, 0.1]
        final_data["rpm"]["desired_levels"]=desired_levels
        rpm_prob_list_list=[]
        rpm_save_list_list=[]
        rpm_volume_list_list=[]
        rpm_indices_list_list=[]
        rpm_check_thres_list_list=[]
        final_data["rpm"]["viz_timesteps"] = args.viz_timesteps
        for ti in args.viz_timesteps:
            rho_min_list = data[ti]["dens_min_list"]
            rho_max_list = data[ti]["dens_max_list"]
            prob_min_list = data[ti]["prob_min_list"]
            prob_max_list = data[ti]["prob_max_list"]
            vertice_list = data[ti]["vertice_list"]
            volume_list = data[ti]["volume_list"]

            final_data["rpm"][ti]={}
            final_data["rpm"][ti]["dens_min_list"] = rho_min_list
            final_data["rpm"][ti]["dens_max_list"] = rho_max_list
            final_data["rpm"][ti]["prob_min_list"] = prob_min_list
            final_data["rpm"][ti]["prob_max_list"] = prob_max_list
            final_data["rpm"][ti]["vertice_list"] = vertice_list
            final_data["rpm"][ti]["volume_list"] = volume_list

            if args.sim_vol:
                delaunay_list = [Delaunay(vert_item) for vert_item in vertice_list]

            volume_t_alpha_list.append([])
            rho_min_clip = 0.0
            prob_minimum = np.min(prob_min_list)
            prob_maximum = np.max(prob_max_list)
            prob_min = min(rho_min_clip, prob_minimum)
            prob_max = max(rho_min_clip, prob_maximum)

            exclude_indices = []
            if args.select_by_dens:
                indices = np.argsort(rho_max_list)
            else:
                indices = np.argsort(prob_max_list)
            total_volume = np.sum([volume_list[i] for i in range(len(prob_max_list))])
            prev_prob=1.0
            rpm_prob_list_list.append([prev_prob])
            rpm_save_list_list.append([0.0])
            rpm_volume_list_list.append([total_volume])
            rpm_indices_list_list.append([indices])
            rpm_check_thres_list_list.append([0.0])

            for i in indices:
                exclude_indices.append(i)
                # include_indices = list(set(indices) - set(exclude_indices))
                include_indices = [iii for iii in indices]
                for del_i in exclude_indices:
                    include_indices.remove(del_i)

                prob = np.sum([prob_min_list[i] for i in include_indices]) / (
                        np.sum([prob_min_list[i] for i in include_indices]) + np.sum(
                    [prob_max_list[i] for i in exclude_indices])
                )
                exclude_volume = np.sum([volume_list[i] for i in exclude_indices])
                ratio = exclude_volume / total_volume
                for level in desired_levels:
                    if prev_prob >= level and prob < level:
                        rpm_volume_list_list[-1].append(total_volume * (1 - ratio))
                        rpm_prob_list_list[-1].append(prob)
                        rpm_save_list_list[-1].append(ratio)
                        rpm_indices_list_list[-1].append(include_indices)
                        if args.select_by_dens:
                            rpm_check_thres_list_list[-1].append(rho_max_list[i])
                        else:
                            rpm_check_thres_list_list[-1].append(prob_max_list[i])

                        print("t=%02d  ignored=%d  prob=%.4f  save=%.4f currV:%.9f originalV:%.9f" % (
                            ti, len(exclude_indices), prob, ratio, total_volume - exclude_volume, total_volume))
                        break
                prev_prob = prob
            volume_t_alpha_list[-1].append(total_volume - exclude_volume)
        final_data["rpm"]["rpm_prob_list_list"] = rpm_prob_list_list
        final_data["rpm"]["rpm_save_list_list"] = rpm_save_list_list
        final_data["rpm"]["rpm_volume_list_list"] = rpm_volume_list_list
        final_data["rpm"]["rpm_indices_list_list"] = rpm_indices_list_list
        final_data["rpm"]["rpm_check_thres_list_list"]  = rpm_check_thres_list_list


        np.savez(os.path.join(args.exp_dir_full,"rpm_prob_list_list"), rpm_prob_list_list)
        np.savez(os.path.join(args.exp_dir_full,"rpm_save_list_list"), rpm_save_list_list)
        np.savez(os.path.join(args.exp_dir_full,"rpm_volume_list_list"), rpm_volume_list_list)
        np.savez(os.path.join(args.exp_dir_full,"rpm_indices_list_list"), rpm_indices_list_list)

    np.savez(os.path.join(args.exp_dir_full, "final_data_%s" % (args.exp_mode)), final_data)

    def plot_pts(ti):
        plt.scatter(traj_data[:args.max_sample, ti, 0], traj_data[:args.max_sample, ti, 1], s=1)
        ax = plt.gca()
        ax.set_aspect(1.0 / ((y_max - y_min) / (x_max - x_min)), adjustable='box')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    if args.plot_all:
        from labellines import labelLine, labelLines
        # level_set_list=[0.99, 0.95, 0.90, 0.85, 0.80]
        level_set_list = [0.99, 0.9, 0.80, 0.7, 0.50]
        r_rall_list=[]
        eps_list=[]
        thres_list=[]
        for i,ti in enumerate(args.viz_timesteps):
            r_rall_list.append([])
            eps_list.append([])
            thres_list.append([])
            standard_volume = cvx_volume_list[i]  # rpm_volume_list_list[i][0]

            if not args.skip_most_figs:
                # cvx hull method & volume
                # plt.subplot(2, 2, 1)
                plot_pts(ti)
                # for vert in cvx_vertices_list[i]:
                    # print(vert)
                    # plot_polygon(vert[:,:2])
                plot_polygon([proj_xy(vert, args.x_index, args.y_index) for vert in cvx_vertices_list[i]],fill=False)
                plt.title("ConvexHull vol:%.4fX"%(cvx_volume_list[i]/standard_volume))
                plt.tight_layout()
                plt.savefig(os.path.join(args.viz_dir, "fig1_cvx_%03d.png" % (ti)), bbox_inches='tight', pad_inches=0)
                plt.close()

            if "reachlp" in args.method:
                # bbox method & volume
                # plt.subplot(2, 2, 2)
                if not args.skip_most_figs:
                    plot_pts(ti)
                    ax=plt.gca()
                    for bbox in rlp_bbox_list[i]:
                        # plot_polygon(bbox_2_vert(bbox),fill=False)
                        b_xmin=bbox[args.x_index, 0]
                        b_xmax=bbox[args.x_index, 1]
                        b_ymin = bbox[args.y_index, 0]
                        b_ymax = bbox[args.y_index, 1]
                        rect = patches.Rectangle((b_xmin, b_ymin), (b_xmax-b_xmin), (b_ymax-b_ymin),
                                                 linewidth=1, edgecolor='g', facecolor='none')
                        # Add the patch to the Axes
                        ax.add_patch(rect)

                    plt.title("GSG vol:%.4fX" % (rlp_volume_list[i]/standard_volume))
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.viz_dir, "fig2_rlp_%03d.png" % (ti)), bbox_inches='tight', pad_inches=0)
                    plt.close()

            '''
            rho_min_list = data[ti]["dens_min_list"]
            rho_max_list = data[ti]["dens_max_list"]
            prob_min_list = data[ti]["prob_min_list"]
            prob_max_list = data[ti]["prob_max_list"]
            vertice_list = data[ti]["vertice_list"]
            volume_list = data[ti]["volume_list"]
            '''
            # Ours estimated density & volume (uniform)
            # plt.subplot(2, 2, 3)
            if not args.skip_most_figs:
                for vert in data[ti]["vertice_list"]:
                    plot_polygon(vert, alpha=1.0, color=np.random.rand(3))
                ax = plt.gca()
                ax.set_aspect(1.0 / ((y_max - y_min) / (x_max - x_min)), adjustable='box')
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                plt.title("RPM vol:%.4fX"%(rpm_volume_list_list[i][0]/standard_volume))
                plt.tight_layout()
                plt.savefig(os.path.join(args.viz_dir, "fig3_rpm0_%03d.png" % (ti)), bbox_inches='tight', pad_inches=0)
                plt.close()

            # Ours method & volume (uniform, p>0.9)
            for draw in ["prob", "density"]:
                for l_i, level_set in enumerate(level_set_list):
                    # plt.subplot(2, 2, 4)
                    for p_i, prob in enumerate(rpm_prob_list_list[i]):
                        if p_i>0 and prob <level_set and rpm_prob_list_list[i][p_i-1]>=level_set:
                            if not args.skip_most_figs:
                                draw_background()

                            r_rall_list[-1].append(rpm_volume_list_list[i][p_i-1]/standard_volume)
                            eps_list[-1].append(rpm_prob_list_list[i][p_i - 1])
                            thres_list[-1].append(rpm_check_thres_list_list[i][l_i])
                            if not args.skip_most_figs:
                                if draw =="prob":
                                    prob_min = np.min([data[ti]["prob_min_list"][vert_idx] for vert_idx in rpm_indices_list_list[i][p_i-1]])
                                    prob_max = np.max([data[ti]["prob_max_list"][vert_idx] for vert_idx in rpm_indices_list_list[i][p_i-1]])
                                    sort_indices = np.argsort(data[ti]["prob_max_list"])
                                else:
                                    rho_min = np.min([data[ti]["dens_min_list"][vert_idx] for vert_idx in rpm_indices_list_list[i][p_i-1]])
                                    rho_max = np.max([data[ti]["dens_max_list"][vert_idx] for vert_idx in rpm_indices_list_list[i][p_i-1]])
                                    if ti==0:
                                        rho_min = rho_max / 10.0
                                    sort_indices = np.argsort(data[ti]["dens_max_list"])

                                for vert_idx in sort_indices:
                                    if vert_idx in rpm_indices_list_list[i][p_i-1]:
                                        vert = data[ti]["vertice_list"][vert_idx]
                                        if draw == "prob":
                                            norm_rho = (data[ti]["prob_max_list"][vert_idx] - prob_min) / (np.clip(prob_max - prob_min, a_min=1e-4, a_max=1e10))
                                        else:
                                            norm_rho = (data[ti]["dens_max_list"][vert_idx] - rho_min) / (np.clip(rho_max - rho_min, a_min=1e-4, a_max=1e10))
                                        color_i = cmap(norm_rho)
                                        plot_polygon(vert, color=color_i, alpha=1.0)
                                        # plot_polygon(vert, color="yellow", alpha=1.0)
                            break
                    if not args.skip_most_figs:
                        ax = plt.gca()
                        ax.set_aspect(1.0 / ((y_max - y_min) / (x_max - x_min)), adjustable='box')
                        plt.xlim(x_min, x_max)
                        plt.ylim(y_min, y_max)
                        if draw == "prob":
                            norm = matplotlib.colors.Normalize(vmin=prob_min, vmax=prob_max)
                        else:
                            norm = matplotlib.colors.Normalize(vmin=rho_min, vmax=rho_max)
                        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
                        plt.title("Ours vol:%.4fX (p=%.3f.>%.3f)" % (rpm_volume_list_list[i][p_i-1]/standard_volume, rpm_prob_list_list[i][p_i-1], level_set))
                        plt.tight_layout()
                        if draw == "prob":
                            plt.savefig(os.path.join(args.viz_dir, "fig_rpm_prob%d_%03d.png"%(l_i+1, ti)), bbox_inches='tight', pad_inches=0)
                        else:
                            plt.savefig(os.path.join(args.viz_dir, "fig_rpm_rho%d_%03d.png" % (l_i + 1, ti)),
                                        bbox_inches='tight', pad_inches=0)
                        plt.close()

        # plot curve for R/Rall vs. eps
        colors1=["brown", "red", "salmon", "coral", "peru"]
        # colors2=["navy", "mediumblue", "royalblue", "rebeccapurple","blueviolet"]
        # colors2 = ["lightcoral", "orangered", "lightseagreen", "steelblue", "rebeccapurple"]
        colors2 = ["#e6194B", "#f58231", "#3cb44b", "#4363d8", "#911eb4"]


        for l_i, level_set in enumerate(level_set_list):
            # rpm_volume_list_list[i][p_i-1]/rpm_volume_list_list[i][0],
            # print(r_rall_list)
            plt.plot(args.viz_timesteps, [r_rall_list[ii][l_i] for ii, ti in enumerate(args.viz_timesteps)],
                     label="R/R_all (p>%.4f)"%(level_set), color=colors1[l_i])
            plt.plot(args.viz_timesteps, [eps_list[ii][l_i] for ii, ti in enumerate(args.viz_timesteps)],
                     label="p>%.4f"%( level_set), color=colors2[l_i])
        plt.xlabel("Timestep")
        plt.ylabel("Volume/Probability")
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.savefig(os.path.join(args.viz_dir, "fig_rpm_zcurve_1.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

        for l_i, level_set in enumerate(level_set_list):
            # rpm_volume_list_list[i][p_i-1]/rpm_volume_list_list[i][0],
            # print(r_rall_list)
            plt.plot(args.viz_timesteps, [r_rall_list[ii][l_i] for ii, ti in enumerate(args.viz_timesteps)],
                     label="p>%.2f"%(level_set), color=colors2[l_i])
            # plt.plot(args.viz_timesteps, [thres_list[ii][l_i] for ii, ti in enumerate(args.viz_timesteps)],
            #          label="eps p>%.4f"%( level_set), color=colors2[l_i])
        plt.grid(axis='y',linestyle='--', linewidth=2)
        plt.xlabel("Timestep")
        plt.ylabel("Volume")
        # plt.legend(bbox_to_anchor=(1.05, 1))
        labelLines(plt.gca().get_lines(), zorder=2.5, align=False, fontsize=14)
        plt.savefig(os.path.join(args.viz_dir, "fig0_rpm_curve_%s.png"%(args.exp_mode)), bbox_inches='tight', pad_inches=0)
        plt.close()

        np.savez(os.path.join(args.viz_dir,"eps_rall.npz"), r_rall_list=r_rall_list, eps_list=eps_list, viz_timesteps=args.viz_timesteps,
                 level_set_list=level_set_list)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_mode', type=str, choices= \
        ['robot', 'dint', 'toon', 'car', 'quad', 'gcas', 'acc', 'pend', 'vdp', 'circ'])
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--bag_path", type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--nt', type=int, default=None)
    parser.add_argument('--dt', type=float, default=None)
    parser.add_argument('--s_mins', nargs="+", type=float)
    parser.add_argument('--s_maxs', nargs="+", type=float)

    parser.add_argument('--x_min', type=float, default=-2)
    parser.add_argument('--x_max', type=float, default=2)
    parser.add_argument('--y_min', type=float, default=-2)
    parser.add_argument('--y_max', type=float, default=2)
    parser.add_argument('--nx', type=int, default=81)
    parser.add_argument('--ny', type=int, default=81)
    parser.add_argument('--x_index', type=int, default=0)
    parser.add_argument('--y_index', type=int, default=1)
    parser.add_argument('--x_label', type=str, default='x')
    parser.add_argument('--y_label', type=str, default='y')

    # TODO exp configs
    parser.add_argument('--distribution', type=str, default='uniform', choices=["uniform", "gaussian"])
    parser.add_argument('--mu', type=float, nargs="+", default=None)
    parser.add_argument('--cov', type=float, nargs="+", default=None)

    parser.add_argument('--plot_frs_density', action='store_true')
    parser.add_argument('--plot_frs_prob', action='store_true')
    parser.add_argument('--dens_log_val', action='store_true')
    parser.add_argument('--dens_log_val_minimum', type=float, default=-10.0)

    parser.add_argument('--check_brs', action='store_true')
    parser.add_argument('--bbox_mode', action='store_true')
    parser.add_argument('--q_mins', type=float, nargs="+", default=None)
    parser.add_argument('--q_maxs', type=float, nargs="+", default=None)

    parser.add_argument('--plot_varied_frs', action='store_true')
    parser.add_argument('--viz_timesteps', type=int, nargs="+", default=None)
    parser.add_argument('--alphas', type=float, nargs="+", default=None)
    parser.add_argument('--select_by_dens', action='store_true')

    parser.add_argument('--run_density_measure_exp', action='store_true')
    parser.add_argument('--track_id', type=int, default=None)
    parser.add_argument('--track_radius', type=float, default=None)
    parser.add_argument('--track_type', type=str, choices=["ball", "rect"], default="rect")
    parser.add_argument('--sim_data_path', type=str, default=None)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--n_sampling_points', type=int, default=None)

    parser.add_argument('--run_reach', action='store_true')
    parser.add_argument('--method', type=str, nargs="+", default=None)
    parser.add_argument('--traj_data_path', type=str, default=None)

    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--num_simulations', type=int, default=1e4)
    parser.add_argument("--partitioner", default="GreedySimGuided",
        choices=[
            "None",
            "Uniform",
            "SimGuided",
            "GreedySimGuided",
            "AdaptiveGreedySimGuided",
            "UnGuided",
        ],
    )
    parser.add_argument("--term_type", default="time_budget",
        choices=[
            "time_budget",
            "verify",
            "input_cell_size",
            "num_propagator_calls",
            "pct_improvement",
            "pct_error",
        ],
    )
    parser.add_argument("--term_val", default=2.0, type=float, help="value of condition to terminate (default: 2)")
    parser.add_argument("--interior_condition", default="lower_bnds", choices=["lower_bnds", "linf", "convex_hull"],
        help="type of bound to optimize for (default: lower_bnds)")
    parser.add_argument("--propagator", default="CROWN_LIRPA",
        choices=[
            "IBP",
            "CROWN",
            "CROWN_LIRPA",
            "IBP_LIRPA",
            "CROWN-IBP_LIRPA",
            "FastLin_LIRPA",
            "Exhaustive_LIRPA",
            "SDP",
        ])
    parser.add_argument('--n_mc', type=int, default=100000)
    parser.add_argument('--input_dim', type=int, default=None)
    parser.add_argument('--hiddens', type=int, nargs="+", default=None)
    parser.add_argument('--debug_gap', type=int, default=5)
    parser.add_argument('--debug_start_from', type=int, default=1)
    parser.add_argument('--plot_all', action="store_true")
    parser.add_argument('--max_sample', type=int, default=1000)

    parser.add_argument('--skip_most_figs', action='store_true')

    parser.add_argument('--new_vdp', action='store_true')
    parser.add_argument('--new_vdp2', action='store_true')
    parser.add_argument('--sim_vol', action='store_true')
    parser.add_argument('--advance_plot', action='store_true')
    parser.add_argument('--plot_mesh', action='store_true')
    parser.add_argument('--plot_mesh_prob', action='store_true')
    parser.add_argument('--plot_mesh_log', action='store_true')
    return parser.parse_args()


def main():
    if args.output_path is None:
        if args.bag_path is None:
            if args.new_vdp and args.exp_mode=="vdp":
                suffix="_new"
            if args.new_vdp2 and args.exp_mode=="vdp":
                suffix="_new2"
            else:
                suffix=""
            args.bag_path = "cache/reach_%s%s/bag.npz" % (args.exp_mode, suffix)
        args.output_path = os.path.dirname(args.bag_path)
    # os.makedirs(os.path.join(args.output_path, "viz"), exist_ok=True)

    np.random.seed(1007)

    sub_utils.setup_data_exp_and_logger(args, reach_prob=True)

    if args.viz_timesteps is None:
        args.viz_timesteps = range(args.nt)
    if args.run_reach and ("rpm" not in args.method):
        bag = None
        saved_data = None
    else:
        bag = np.load(args.bag_path, allow_pickle=True)['arr_0']
        bag = update_bag(bag)

        bag, saved_data = prepare_bag_data(bag)
        np.savez("%s/saved_data" % (args.exp_dir_full), saved_data)
        print("Finished preparing bag")

    if args.plot_frs_density or args.plot_frs_prob:
        plot_frs_func(saved_data)
        print("Finished plotting FRS")

    # TODO brs
    if args.check_brs:
        check_brs_func(bag)

    # TODO varied
    if args.plot_varied_frs:
        plot_varied_frs_func(saved_data)

    # TODO density measure exps
    if args.run_density_measure_exp:
        run_density_measure_exp_func(bag, saved_data)

    # TODO reach measure exps
    if args.run_reach:
        run_reach_func(bag, saved_data)



if __name__ =="__main__":
    args = get_args()
    t1 = time.time()
    main()
    print("Finished in %.4f s" % (time.time() - t1))
