import numpy as np
import os
import argparse
from labellines import labelLines
import matplotlib.pyplot as plt
from pypoman import plot_polygon
import matplotlib.patches as patches
import pylab
import pickle
import matplotlib

font = {
    # 'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 18
}
matplotlib.rc('font', **font)


def smart_load_traj(path):
    return np.load(path, allow_pickle=True)['arr_0'].item()

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
    plt.scatter(traj_data[:args.max_sample, ti, 0], traj_data[:args.max_sample, ti, 1], s=1)
    ax = plt.gca()
    ax.set_aspect(1.0 / ((args.y_max - args.y_min) / (args.x_max - args.x_min)), adjustable='box')
    plt.xlim(args.x_min, args.x_max)
    plt.ylim(args.y_min, args.y_max)

def proj_xy(vert, x_index=0, y_index=1):
    return (vert[x_index], vert[y_index])


# TODO configurations
SKIP_MOST=True

level_set_list = [0.99, 0.9, 0.80, 0.7, 0.50]
cmap = matplotlib.cm.inferno

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="")
fut_args = parser.parse_args()

final_data=np.load(fut_args.path, allow_pickle=True)["arr_0"].item()
args = final_data["args"]


traj_data = smart_load_traj("../data/%s_data.npz"%(args.exp_mode))["s_list"]
os.makedirs("cache", exist_ok=True)

with open("logs_data/%s/vols.pkl"%(args.exp_mode), "rb") as f:
     dryvr_vol=pickle.load(f)

r_rall_list=[]
eps_list=[]
thres_list=[]
print("draw figure 2")
for i, ti in enumerate(args.viz_timesteps):
    r_rall_list.append([])
    eps_list.append([])
    thres_list.append([])
    standard_volume = final_data["cvx"]["cvx_volume_list"][i]

    # plot cvx
    if not SKIP_MOST:
        plot_pts(ti)
        plot_polygon([proj_xy(vert, args.x_index, args.y_index)
                      for vert in final_data["cvx"]["cvx_vertices_list"][i]], fill=False)
        plt.title("ConvexHull vol:%.4fX"%(final_data["cvx"]["cvx_volume_list"][i]/standard_volume))
        plt.tight_layout()
        plt.savefig(os.path.join("cache",
            "%s_1_cvx_%03d.png" % (args.exp_mode, ti)), bbox_inches='tight', pad_inches=0)
        plt.close()


    # GSG
    if not SKIP_MOST:
        plot_pts(ti)
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

        plt.title("GSG vol:%.4fX" % (final_data["reachlp"]["rlp_volume_list"][i] / standard_volume))
        plt.tight_layout()
        plt.savefig(os.path.join("cache", "%s_2_gsg_%03d.png" % (args.exp_mode, ti)), bbox_inches='tight', pad_inches=0)
        plt.close()

    # TODO
    # if not SKIP_MOST:
    #     plot_pts(ti)
    #     rect = patches.Circle((b_xmin, b_ymin), (b_xmax - b_xmin), (b_ymax - b_ymin),
    #                              linewidth=1, edgecolor='g', facecolor='none')
    #     # Add the patch to the Axes
    #     ax.add_patch(rect)
    #     plt.title("DryVR vol:%.4fX" % (dryvr_vol[ti] / standard_volume))
    #     plt.tight_layout()
    #     plt.savefig(os.path.join("cache",
    #                              "%s_3_dryvr_%03d.png" % (args.exp_mode, ti)), bbox_inches='tight', pad_inches=0)
    #     plt.close()

    if not SKIP_MOST:
        # RPM
        for vert in final_data["rpm"][ti]["vertice_list"]:
            plot_polygon(vert, alpha=1.0, color=np.random.rand(3))
        ax = plt.gca()
        ax.set_aspect(1.0 / ((args.y_max - args.y_min) / (args.x_max - args.x_min)), adjustable='box')
        plt.xlim(args.x_min, args.x_max)
        plt.ylim(args.y_min, args.y_max)
        plt.title("RPM vol:%.4fX" % (final_data["rpm"]["rpm_volume_list_list"][i][0] / standard_volume))
        plt.tight_layout()
        plt.savefig(os.path.join("cache", "%s_2_rpm0_%03d.png" % (args.exp_mode, ti)), bbox_inches='tight', pad_inches=0)
        plt.close()


    # RPM (probablistic)
    rpm_prob_list_list = final_data["rpm"]["rpm_prob_list_list"]
    rpm_volume_list_list = final_data["rpm"]["rpm_volume_list_list"]
    rpm_check_thres_list_list = final_data["rpm"]["rpm_check_thres_list_list"]
    rpm_indices_list_list = final_data["rpm"]["rpm_indices_list_list"]

    data = final_data["rpm"]

    for draw in ["prob", "density"]:
        for l_i, level_set in enumerate(level_set_list):
            for p_i, prob in enumerate(rpm_prob_list_list[i]):
                if p_i > 0 and prob < level_set and rpm_prob_list_list[i][p_i - 1] >= level_set:
                    if not SKIP_MOST:
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

                    if not SKIP_MOST:
                        for vert_idx in sort_indices:
                            if vert_idx in rpm_indices_list_list[i][p_i - 1]:
                                vert = data[ti]["vertice_list"][vert_idx]
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
                plt.title("Ours vol:%.4fX (p=%.3f.>%.3f)" % (rpm_volume_list_list[i][p_i - 1] / standard_volume, rpm_prob_list_list[i][p_i - 1], level_set))
                plt.tight_layout()
                if draw == "prob":
                    plt.savefig(os.path.join("cache", "%s_4_rpm_prob%d_%03d.png" % (args.exp_mode, l_i + 1, ti)),
                                bbox_inches='tight', pad_inches=0)
                else:
                    plt.savefig(os.path.join("cache", "%s_4_rpm_rho%d_%03d.png" % (args.exp_mode, l_i + 1, ti)),
                                bbox_inches='tight', pad_inches=0)
                plt.close()


from mpl_toolkits.axes_grid1 import make_axes_locatable


if args.exp_mode=="robot":
    args.viz_timesteps = args.viz_timesteps[:-1]

print("draw figure 3")
# finish t-wise ploting now figure 3
colors2 = ["#e6194B", "#f58231", "#3cb44b", "#4363d8", "#911eb4"]


axMain = plt.subplot(111)


high_high = max(np.max([final_data["reachlp"]["rlp_volume_list"][ii]/standard_volume for ii, ti in enumerate(args.viz_timesteps)]), np.max([dryvr_vol[ti]/standard_volume for ii, ti in enumerate(args.viz_timesteps)]))
peak=0
lowest = min(np.min([final_data["reachlp"]["rlp_volume_list"][ii]/standard_volume for ii, ti in enumerate(args.viz_timesteps)]), np.min([dryvr_vol[ti]/standard_volume for ii, ti in enumerate(args.viz_timesteps)]))
if lowest < peak:
    peak = lowest
# RPM's curves
for l_i, level_set in enumerate(level_set_list):
    axMain.plot(args.viz_timesteps, [r_rall_list[ii][l_i] for ii, ti in enumerate(args.viz_timesteps)],
             label="p>%.2f" % (level_set), color=colors2[l_i])
    peak = max(peak, np.max([r_rall_list[ii][l_i] for ii, ti in enumerate(args.viz_timesteps)]))

if args.exp_mode=="car":
    peak+=0.3
axMain.set_yscale('linear')
axMain.set_ylim((0.00, peak))
axMain.grid(axis='y', linestyle='--', linewidth=2)
labelLines(plt.gca().get_lines(), zorder=2.5, align=False, fontsize=16)
plt.xlabel("Timestep")
plt.ylabel("Volume")
divider = make_axes_locatable(axMain)
axLog = divider.append_axes("top", size=1.0, pad=0.02, sharex=axMain)

# GSG Curve
axLog.set_yscale('log')
axLog.set_ylim((lowest*0.8, high_high*1.5))
axLog.plot(args.viz_timesteps, [final_data["reachlp"]["rlp_volume_list"][ii]/standard_volume for ii, ti in enumerate(args.viz_timesteps)],
             label="GSG", color="green")
# DryVR Curve
axLog.plot(args.viz_timesteps, [dryvr_vol[ti]/standard_volume for ii, ti in enumerate(args.viz_timesteps)],
             label="DryVR", color="red")


labelLines(plt.gca().get_lines(), zorder=2.5, align=False, fontsize=16)
plt.savefig(os.path.join("cache", "%s_5_rpm_curve_%s.png"%(args.exp_mode, args.exp_mode)), bbox_inches='tight', pad_inches=0.1)
plt.close()
