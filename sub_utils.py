import os, sys
import shutil
import numpy as np
import torch
from scipy.io import savemat
import gurobipy as gp
from gurobipy import GRB
import cdd
from scipy.spatial import ConvexHull
import scipy
from scipy.stats import multivariate_normal
from datetime import datetime
import time
from os.path import join as ospj

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.history.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    def __init__(self):
        self._terminal = sys.stdout
        self._timestr = datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S")

    def create_log(self, log_path):
        self.log = open(log_path + "/log-%s.txt" % self._timestr, "a", 1)

    def write(self, message):
        self._terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def write_cmd_to_file(log_dir, argv):
    with open(ospj(log_dir, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(argv))

def setup_data_exp_and_logger(args, train_nn=False, reach_prob=False):
    logger = Logger()
    sys.stdout = logger
    if train_nn:
        EXP_ROOT_DIR = "cache"
        exp_dir_full = os.path.join(EXP_ROOT_DIR, "g%s_%s_%s" % (logger._timestr, args.exp_mode, args.exp_name))
    elif reach_prob:
        EXP_ROOT_DIR = args.output_path
        exp_dir_full = os.path.join(EXP_ROOT_DIR, "g%s%s" % (logger._timestr, args.exp_name))
    else:
        EXP_ROOT_DIR = "cache"
        exp_dir_full = os.path.join(EXP_ROOT_DIR, "g%s_%s_%s" % (logger._timestr, args.exp_mode, args.exp_name))
    args.exp_dir_full = exp_dir_full
    args.viz_dir = os.path.join(exp_dir_full, "viz")
    args.bak_dir = os.path.join(exp_dir_full, "src")
    os.makedirs(args.viz_dir, exist_ok=True)
    os.makedirs(args.bak_dir, exist_ok=True)

    if train_nn:
        args.model_dir = os.path.join(exp_dir_full, "models")
        os.makedirs(args.model_dir, exist_ok=True)

    logger.create_log(exp_dir_full)
    write_cmd_to_file(exp_dir_full, sys.argv)

    for fname in os.listdir('./'):
        if fname.endswith('.py'):
            shutil.copy(fname, os.path.join(args.bak_dir, fname))
    np.savez(os.path.join(exp_dir_full, 'args'), args=args)
    return args

def save_model_in_julia_format(model_path, save_path, in_dim, out_dim, args):
    checkpoint = torch.load(model_path)
    keys = list(checkpoint.keys())
    num_layers = len(keys) // 2

    # convert to julia model format
    if args.normalize:
        mdic = {"weights": [], "biases": [],
                "X_mean": args.in_means, "X_std": args.in_stds,
                "Y_mean": args.out_means, "Y_std": args.out_stds}
    else:
        mdic = {"weights": [], "biases": [],
                "X_mean": np.zeros(in_dim), "X_std": np.ones(in_dim),
                "Y_mean": np.zeros(out_dim), "Y_std": np.ones(out_dim)}

    for i in range(num_layers):
        ori_w = checkpoint[keys[i * 2]]
        ori_b = checkpoint[keys[i * 2 + 1]]
        mdic["weights"].append(ori_w.detach().cpu().numpy())
        mdic["biases"].append(ori_b.detach().cpu().numpy())
    mdic["weights"] = np.array(mdic["weights"], dtype=object)
    mdic["biases"] = np.array(mdic["biases"], dtype=object)
    savemat(save_path, mdic)


def get_data_stat(raw_data, train_num_traj, use_log=False, include_minmax=False):
    # TODO check valid stat 0<...<1e4
    in_x_means = np.mean(raw_data['s_list'][:train_num_traj, 0, :], axis=0)
    in_x_stds = np.std(raw_data['s_list'][:train_num_traj, 0, :], axis=0)
    in_t_means = np.zeros((1,))
    in_t_stds = np.ones((1,))

    out_rho_means = np.zeros((1,))
    if use_log:
        out_rho_stds = np.ones((1,)) * np.max(np.abs(np.log(raw_data['rho_list'][:train_num_traj, :])), axis=(0, 1))
    else:
        out_rho_stds = np.ones((1,)) * np.max(np.abs(raw_data['rho_list'][:train_num_traj, :]), axis=(0, 1))
    out_x_means = np.mean(raw_data['s_list'][:train_num_traj, :, :], axis=(0, 1))
    out_x_stds = np.std(raw_data['s_list'][:train_num_traj, :, :], axis=(0, 1))

    in_means = np.concatenate((in_x_means, in_t_means))
    in_stds = np.concatenate((in_x_stds, in_t_stds))
    out_means = np.concatenate((out_rho_means, out_x_means))
    out_stds = np.concatenate((out_rho_stds, out_x_stds))

    if include_minmax:
        in_x_mins = np.min(raw_data['s_list'][:train_num_traj, 0, :], axis=0)
        in_x_maxs = np.max(raw_data['s_list'][:train_num_traj, 0, :], axis=0)
        in_t_mins = np.ones((1,)) * np.min(raw_data['t_list'][:train_num_traj, :])
        in_t_maxs = np.ones((1,)) * np.max(raw_data['t_list'][:train_num_traj, :])
        out_x_mins = np.min(raw_data['s_list'][:train_num_traj, :, :], axis=(0,1))
        out_x_maxs = np.max(raw_data['s_list'][:train_num_traj, :, :], axis=(0,1))
        if use_log:
            out_rho_mins= np.ones((1,)) * np.min(np.log(raw_data['rho_list'][:train_num_traj, :]))
            out_rho_maxs = np.ones((1,)) * np.max(np.log(raw_data['rho_list'][:train_num_traj, :]))
        else:
            out_rho_mins = np.ones((1,)) * np.min(raw_data['rho_list'][:train_num_traj, :])
            out_rho_maxs = np.ones((1,)) * np.max(raw_data['rho_list'][:train_num_traj, :])

        in_min = np.concatenate((in_x_mins, in_t_mins))
        in_max = np.concatenate((in_x_maxs, in_t_maxs))
        out_min = np.concatenate((out_rho_mins, out_x_mins))
        out_max = np.concatenate((out_rho_maxs, out_x_maxs))

        return in_means, in_stds, out_means, out_stds, in_min, in_max, out_min, out_max
    else:
        return in_means, in_stds, out_means, out_stds

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

def convert_fig(delay, img_path, gif_path):
    "convert -delay 25 -loop 0 est*.png ani_est.gif"
    print("convert -delay %d -loop 0 %s %s/ani_est.gif"%(delay, img_path, gif_path))
    os.system("convert -delay %d -loop 0 %s %s/ani_est.gif"%(delay, img_path, gif_path))

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
def convert_multiple_gifs(img_paths, labels, output_dir, nt, lag):
    imgs_list = [[Image.open("%s/%s" % (path, img_fname % (i))) for i in range(nt)]
                 for path, img_fname in img_paths]

    text_pad = 25
    font_size = 36
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", font_size)

    n_imgs = len(imgs_list)
    output_dir_full = "%s/cache_gif"%(output_dir)
    if os.path.exists(output_dir_full):
        shutil.rmtree(output_dir_full, ignore_errors=True)
    os.makedirs(output_dir_full)
    width=0
    height=0
    for i in range(n_imgs):
        for j in range(nt):
            width = max(imgs_list[i][j].width, width)
            height = max(imgs_list[i][j].height, height)
    font_color = (255, 255, 255)

    frames=[]
    for j in range(nt):
        new_frame = Image.new('RGB', (width * n_imgs, height + 2 * text_pad))
        for i in range(n_imgs):
            frame=imgs_list[i][j].convert("RGB")
            new_frame.paste(frame,
                            (width*i, 0, width*i+frame.width, frame.height))
            ImageDraw.Draw(
                new_frame  # Image
            ).text(
                (width*i, height +0.1*text_pad),  # Coordinates
                '%s (t=%d)' % (labels[i], j),  # Text
                font_color,  # Color
                font=font,
            )
        frames.append(new_frame)
        new_frame.save("%s/%03d.png" % (output_dir_full, j))
    os.system("convert -delay %d -loop 0 %s/*.png %s/ani.gif" % (lag, output_dir_full, output_dir_full))
    os.system("cp %s/ani.gif %s/../ani.gif" % (output_dir_full, output_dir_full))



# load A,b,c,d
# load possible init, end conditions
# for a given query (xt, t), know its
def non_empty(A, b):
    cdim, xdim = A.shape

    m = gp.Model("lp")
    m.setParam('OutputFlag', 0)
    m.params.threads = 1
    xs = m.addVars(list(range(xdim)), name="x", lb=float('-inf'), ub=float('inf'))
    m.setObjective(xs[0], GRB.MINIMIZE)
    for j in range(cdim):
        m.addConstr(gp.quicksum(xs[i]*A[j, i] for i in range(xdim)) <= b[j], "c%d"%j)
    m.optimize()
    if m.status == GRB.INF_OR_UNBD:
        m.setParam(GRB.Param.Presolve, 0)
        m.optimize()

    if m.status == GRB.OPTIMAL:
        rho_min = m.objVal

        m = gp.Model("lp")
        m.setParam('OutputFlag', 0)
        m.params.threads = 1
        xs = m.addVars(list(range(xdim)), name="x", lb=float('-inf'), ub=float('inf'))
        m.setObjective(1.0*xs[0], GRB.MAXIMIZE)
        for j in range(cdim):
            m.addConstr(gp.quicksum(xs[i] * A[j, i] for i in range(xdim)) <= b[j], "c%d" % j)
        m.optimize()
        if m.status == GRB.INF_OR_UNBD:
            m.setParam(GRB.Param.Presolve, 0)
            m.optimize()

        rho_max = m.objVal

        return True, rho_min, rho_max
    elif m.status == GRB.INFEASIBLE:
        return False, None, None
    else:
        print("Un-recognized status: %s" % (m.status))
        raise NotImplementedError


def get_A_b_vertices(A, b):
    assert len(b.shape)==1
    b_2d = b.reshape((b.shape[0], 1))
    linsys = cdd.Matrix(np.hstack([b_2d, -A]), number_type='float')
    linsys.rep_type = cdd.RepType.INEQUALITY
    P = cdd.Polyhedron(linsys)
    generators = P.get_generators()
    return np.array(generators)


def get_A_b_vertices_robust(A, b, ti, i, thres=1e4):
    # get the vertices and the volume for the output
    try:
        vertices= get_A_b_vertices(A, b)
        return vertices
    except RuntimeError:
        print("Error happens", ti, i)
        print(A, b)
        # new_A=np.array(A)
        # new_b=np.array(b)
        for dbg_i in range(A.shape[0]):  # TODO Row check for A
            check_val = np.max(np.abs(A[dbg_i]))
            if check_val > thres:
                print("the %d-th row too large, divided by %.4f~" % (dbg_i, check_val))
                # new_A[dbg_i] = A[dbg_i] / check_val
                # new_b[dbg_i] = b[dbg_i] / check_val
                A[dbg_i] = A[dbg_i] / check_val
                b[dbg_i] = b[dbg_i] / check_val
        try:
            vertices = get_A_b_vertices(A, b)
            return vertices
        except RuntimeError:
            print("Cannot solve for", ti, i, "return None")
            return None


def get_A_b_sys_vertices_volume_robust(vertices, is_input=False):
    # status=0, no vertices
    # status=1, verified
    # status=2, len(vertices)<dim+1
    # status=3, co-planar
    sys_vertices = None
    volume = None
    if len(vertices) > 0:
        dim = len(vertices[0])
        if len(vertices) < dim:  # needs k+1 vertices in k dim-space to make volume>0
            status=2
        else:
            try:
                if is_input:
                    sys_vertices = vertices[:, 1: -1]
                else:
                    sys_vertices = vertices[:, 1 + 1:]
                cvx_hull = ConvexHull(points=sys_vertices)  # remove the indicator dim and the first dim (density gain)
                volume = cvx_hull.volume
                status = 1
            except scipy.spatial.qhull.QhullError:
                print("simplex V=0, just remove it")
                status = 3
    else:
        status=0
    return sys_vertices, volume, status

def gaussian_pdf(x, mu, cov):
    return multivariate_normal.pdf(x, mean=mu, cov=cov)