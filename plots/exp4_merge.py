import numpy as np
import os
import argparse
from os.path import join as ospj
from labellines import labelLine, labelLines
import matplotlib.pyplot as plt
from pypoman import plot_polygon, project_polytope
import matplotlib
import matplotlib.patches as patches
import pylab
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches


ell_centers=np.array([[-0.00760233,  0.00931574],
[-0.00484125,  0.01372301] ,
[0.00416184, 0.02331915] ,
[0.01788187 ,0.03205321] ,
[0.03518922, 0.03661552] ])

ell_radius=[3.606496396974743,
4.37094728052623,
5.186323816296369,
5.612770789981489,
5.713565858482912]

font = {
    # 'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 16
}
matplotlib.rc('font', **font)

def smart_load_traj(path):
    if ".npz" not in path:
        the_path = os.path.join("../..", 'data',path, 'traj_data.npz')
    else:
        the_path = path
    return np.load(the_path, allow_pickle=True)['arr_0'].item()

def draw_background():
    ax = pylab.gca()
    patch = pylab.Polygon([[args.x_min, args.y_min],
                           [args.x_max, args.y_min],
                           [args.x_max, args.y_max],
                           [args.x_min, args.y_max],
                           [args.x_min, args.y_min]], alpha=1.0, color="black",
                          linestyle="solid", fill=True, linewidth=None)
    ax.add_patch(patch)

def plot_pts(ti):
    plt.scatter(traj_data[:args.max_sample, ti, 0], traj_data[:args.max_sample, ti, 1], s=2, color="gray")
    ax = plt.gca()
    ax.set_aspect(1.0 / ((args.y_max - args.y_min) / (args.x_max - args.x_min)), adjustable='box')
    plt.xlim(args.x_min, args.x_max)
    plt.ylim(args.y_min, args.y_max)

def proj_xy(vert, x_index=0, y_index=1):
    return (vert[x_index], vert[y_index])


# TODO configurations
SKIP_MOST=False

level_set_list = [0.99, 0.9, 0.80, 0.7, 0.50]
cmap = matplotlib.cm.inferno

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="")
fut_args = parser.parse_args()

final_data=np.load(fut_args.path, allow_pickle=True)["arr_0"].item()
args = final_data["args"]

# traj_data=smart_load_traj(args.traj_data_path)["s_list"]
traj_data=smart_load_traj("../data/traj_data.npz")["s_list"]

os.makedirs("cache", exist_ok=True)

with open("logs_data/%s/vols.pkl"%(args.exp_mode), "rb") as f:
     dryvr_vol=pickle.load(f)

r_rall_list=[]
eps_list=[]
thres_list=[]
print("draw figure 2")
jj=0

PLOT_RPM=True
PLOT_IN_HEATMAP=True
INCLUDE_RPM=True

for i, ti in enumerate(args.viz_timesteps):
    r_rall_list.append([])
    eps_list.append([])
    thres_list.append([])
    standard_volume = final_data["cvx"]["cvx_volume_list"][i]

    # GSG
    if not SKIP_MOST:
        # bbox
        if ti in [5,10,20,30]:
            ax = plt.gca()
            for bbox in final_data["reachlp"]["rlp_bbox_list"][i]:
                # plot_polygon(bbox_2_vert(bbox),fill=False)
                b_xmin = bbox[args.x_index, 0]
                b_xmax = bbox[args.x_index, 1]
                b_ymin = bbox[args.y_index, 0]
                b_ymax = bbox[args.y_index, 1]
                rect = patches.Rectangle((b_xmin, b_ymin), (b_xmax - b_xmin), (b_ymax - b_ymin),
                                         linewidth=1, edgecolor='g', facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)


            # cvx
            plot_polygon([proj_xy(vert, args.x_index, args.y_index)
                          for vert in final_data["cvx"]["cvx_vertices_list"][i]], fill=False, color="blue", linewidth=5, alpha=1.0)
            plot_pts(ti)


            # dryVR
            # radius = np.sqrt(dryvr_vol[ti]/np.pi)
            ell = patches.Ellipse((ell_centers[jj,0], ell_centers[jj,1]), ell_radius[jj]*2, ell_radius[jj]*2, linewidth=3, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(ell)

            p1 = mpatches.Patch(color='blue', label='Convex Hull: V=1.00X')

            p2 = mpatches.Patch(color='red', label='DryVR: V=%.2fX'%(dryvr_vol[ti]/standard_volume))

            p3 = mpatches.Patch(color='green', label='GSG: V=%.2fX'%(final_data["reachlp"]["rlp_volume_list"][i] / standard_volume))

            plt.legend(handles=[p1,p2,p3],loc='lower center' , bbox_to_anchor=(0.5, 0.0), borderaxespad=0., prop={'size': 14})



            # plt.title("GSG vol:%.4fX" % (final_data["reachlp"]["rlp_volume_list"][i] / standard_volume))
            plt.tight_layout()
            plt.savefig(os.path.join("cache", "%s_2_merge_%03d.png" % (args.exp_mode, ti)), bbox_inches='tight', pad_inches=0)
            plt.close()
            jj+=1

    if PLOT_RPM:
        # if not SKIP_MOST:
        #     # RPM
        #     for vert in final_data["rpm"][ti]["vertice_list"]:
        #         plot_polygon(vert, alpha=1.0, color=np.random.rand(3))
        #     ax = plt.gca()
        #     ax.set_aspect(1.0 / ((args.y_max - args.y_min) / (args.x_max - args.x_min)), adjustable='box')
        #     plt.xlim(args.x_min, args.x_max)
        #     plt.ylim(args.y_min, args.y_max)
        #     plt.title("RPM vol:%.4fX" % (final_data["rpm"]["rpm_volume_list_list"][i][0] / standard_volume))
        #     plt.tight_layout()
        #     plt.savefig(os.path.join("cache", "%s_2_rpm0_%03d.png" % (args.exp_mode, ti)), bbox_inches='tight', pad_inches=0)
        #     plt.close()


        # RPM (probablistic)
        rpm_prob_list_list = final_data["rpm"]["rpm_prob_list_list"]
        rpm_volume_list_list = final_data["rpm"]["rpm_volume_list_list"]
        rpm_check_thres_list_list = final_data["rpm"]["rpm_check_thres_list_list"]
        rpm_indices_list_list = final_data["rpm"]["rpm_indices_list_list"]

        data = final_data["rpm"]

        for draw in ["density"]:
            for l_i, level_set in enumerate(level_set_list):
                if l_i!=2:
                    continue
                for p_i, prob in enumerate(rpm_prob_list_list[i]):
                    if p_i > 0 and prob < level_set and rpm_prob_list_list[i][p_i - 1] >= level_set:
                        if not SKIP_MOST:
                            if PLOT_IN_HEATMAP:
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

                        tmp_vertes=[]
                        if not SKIP_MOST:
                            for vert_idx in sort_indices:
                                vert = data[ti]["vertice_list"][vert_idx]
                                if vert_idx in rpm_indices_list_list[i][p_i - 1]:
                                    if PLOT_IN_HEATMAP:
                                        if draw == "prob":
                                            norm_rho = (data[ti]["prob_max_list"][vert_idx] - prob_min) / (
                                                np.clip(prob_max - prob_min, a_min=1e-4, a_max=1e10))
                                        else:
                                            if rho_max==rho_min:
                                                norm_rho = (data[ti]["dens_max_list"][vert_idx] - 0) / (
                                                    np.clip(rho_max - 0, a_min=1e-4, a_max=1e10))
                                            else:
                                                norm_rho = (data[ti]["dens_max_list"][vert_idx] - rho_min) / (
                                                    np.clip(rho_max - rho_min, a_min=1e-4, a_max=1e10))
                                        color_i = cmap(norm_rho)
                                        plot_polygon(vert, color=color_i, alpha=1.0)
                                    else:  # PLOT in non HEATMAP
                                        plot_polygon(vert, alpha=1.0, color=np.random.rand(3))

                                # INCLUDE others
                                else:
                                    if INCLUDE_RPM:
                                        tmp_vertes.append(vert)

                            if INCLUDE_RPM:
                                for vert in tmp_vertes:
                                    if PLOT_IN_HEATMAP:
                                        plot_polygon(vert, alpha=1.0, color="limegreen", fill=False)
                                    else:
                                        plot_polygon(vert, alpha=1.0, color=np.random.rand(3), fill=False)

                        break
                if not SKIP_MOST:
                    ax = plt.gca()
                    ax.set_aspect(1.0 / ((args.y_max - args.y_min) / (args.x_max - args.x_min)), adjustable='box')
                    plt.xlim(args.x_min, args.x_max)
                    plt.ylim(args.y_min, args.y_max)

                    if draw == "prob":
                        norm = matplotlib.colors.Normalize(vmin=prob_min, vmax=prob_max)
                    else:
                        norm = matplotlib.colors.Normalize(vmin=rho_min, vmax=rho_max)

                    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
                    plt.title("Ours vol:%.2fX (p=%.2f.>%.2f)" % (rpm_volume_list_list[i][p_i - 1] / standard_volume, rpm_prob_list_list[i][p_i - 1], level_set))
                    plt.tight_layout()
                    if PLOT_IN_HEATMAP:
                        if INCLUDE_RPM:
                            cat_str="heat_with"
                        else:
                            cat_str="heat_no"
                    else:
                        if INCLUDE_RPM:
                            cat_str="plain_with"
                        else:
                            cat_str="plain_no"

                    if ti in [5,10,20,30]:
                        if draw == "prob":
                            plt.savefig(os.path.join("cache", "%s_4_rpm_prob%d_%s_%03d.png" % (args.exp_mode, l_i + 1, cat_str, ti)),
                                        bbox_inches='tight', pad_inches=0)
                        else:
                            plt.savefig(os.path.join("cache", "%s_4_rpm_rho%d_%s_%03d.png" % (args.exp_mode, l_i + 1, cat_str, ti)),
                                        bbox_inches='tight', pad_inches=0)
                    plt.close()
