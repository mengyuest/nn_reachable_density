import os, sys
import time
from os.path import join as ospj
from datetime import datetime
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
from shutil import copyfile
sys.path.append("../")
import utils
from torch.utils.tensorboard import SummaryWriter

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.relu = nn.ReLU()
        self.args = args

        self.linear_list = nn.ModuleList()
        input_dim = 4  # x,y,th,v
        output_dim = 2  # omega, accel
        self.linear_list.append(nn.Linear(input_dim, args.hiddens[0]))
        for i,hidden in enumerate(args.hiddens):
            if i==len(args.hiddens)-1:  # last layer
                self.linear_list.append(nn.Linear(args.hiddens[i], output_dim))
            else:  # middle layers
                self.linear_list.append(nn.Linear(args.hiddens[i], args.hiddens[i + 1]))

    def forward(self, x):
        for i, hidden in enumerate(self.args.hiddens):
            x = self.relu(self.linear_list[i](x))
        x = self.linear_list[len(self.args.hiddens)](x)
        if self.args.output_type=="tanh":
            x = nn.Tanh()(x) * self.args.tanh_gain
        return x

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--viz_freq', type=int, default=10)
    parser.add_argument('--t_len', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.1)  # discrete time for neural network
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=1007)
    parser.add_argument('--exp_name', type=str, default="exp")
    parser.add_argument('--pretrained_path', type=str, default=None)  # TODO
    parser.add_argument('--hiddens', type=int, nargs="+", default=[16, 16])

    parser.add_argument('--obs_weight', type=float, default=1.0)
    parser.add_argument('--dst_weight', type=float, default=1.0)
    parser.add_argument('--reg_weight', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)

    parser.add_argument('--obs_r', type=float, default=1.0)
    parser.add_argument('--xy_range', type=float, default=0.2)

    parser.add_argument('--v_clip', type=float, default=None)
    parser.add_argument('--tanh_gain', type=float, default=None)
    parser.add_argument('--output_type', type=str, default=None)

    parser.add_argument('--show_stat', action='store_true', default=None)

    return parser.parse_args()

def random_state(batch_size, init_xmin, init_xmax, init_ymin, init_ymax, init_thmin, init_thmax, init_vmin, init_vmax):
    state = np.random.rand(batch_size, 4)
    state[:, 0] = state[:, 0] * (init_xmax - init_xmin) + init_xmin
    state[:, 1] = state[:, 1] * (init_ymax - init_ymin) + init_ymin
    state[:, 2] = state[:, 2] * (init_thmax - init_thmin) + init_thmin
    state[:, 3] = state[:, 3] * (init_vmax - init_vmin) + init_vmin
    return state


def dynamics(states, u):
    x, y, th, v = torch.split(states, [1,1,1,1], dim=-1)
    dx = v * torch.cos(th)
    dy = v * torch.sin(th)
    dth = u[:, 0:1]
    dv = u[:, 1:2]
    dsdt = torch.cat([dx, dy, dth, dv], dim=-1)
    return dsdt


def collect_data(actor, state, num_steps, dt, args):
    state_list=[]
    u_list=[]
    for t in range(num_steps):
        u = actor(state.detach())
        state_list.append(state)
        u_list.append(u)
        dsdt = dynamics(state, u)
        state = state + dsdt * dt
        if args.v_clip is not None:
            s_x,s_y,s_th,s_v = torch.split(state, [1,1,1,1], dim=-1)
            s_v = torch.clamp(s_v, -args.v_clip, args.v_clip)
            state = torch.cat([s_x,s_y,s_th,s_v], dim=-1)
    state_list.append(state)
    return torch.stack(state_list, dim=1), torch.stack(u_list, dim=1)


def compute_losses(state_list, u_list, obs, dst, args):

    obs_x, obs_y, obs_r = obs
    dst_x, dst_y, dst_r = dst

    x, y, th, tv = torch.split(state_list, [1, 1, 1, 1], -1)

    obs_dist = ((x - obs_x) ** 2 + (y - obs_y) ** 2) ** 0.5
    dst_dist = ((x - dst_x) ** 2 + (y - dst_y) ** 2) ** 0.5
    safe_mask = obs_dist > obs_r
    goal_mask = dst_dist < dst_r

    last = int(args.t_len * 0.5)

    loss_obs = torch.mean(torch.maximum(obs_r - obs_dist, torch.zeros_like(obs_dist)) * (~safe_mask))
    loss_dst = torch.mean(dst_dist[:,-last:])

    loss_reg = torch.mean(u_list ** 2)

    loss_obs = loss_obs * args.obs_weight
    loss_dst = loss_dst * args.dst_weight
    loss_reg = loss_reg * args.reg_weight
    loss = loss_obs + loss_dst + loss_reg
    return loss_obs, loss_dst, loss_reg, loss, safe_mask, goal_mask

def compute_metrics(safe_mask, goal_mask):
    # print(goal_mask.shape)
    t_len = goal_mask.shape[1]
    last = int(t_len * 0.2)
    safe_rate = torch.mean(1.0 * safe_mask)

    reached_goal = torch.sum(1.0 * goal_mask[:, -last:], dim=1)>=0.5*last
    goal_rate = torch.mean(1.0 * reached_goal)
    # print()
    # print(goal_mask.shape)
    # print(goal_mask)
    # print(reached_goal)
    # print(goal_rate)
    # print()

    return safe_rate, goal_rate

def main():
    args = get_args()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # setup exp dir
    exp_root = utils.get_exp_dir()
    logger = utils.Logger()
    exp_dirname = "g%s_%s" % (logger._timestr, args.exp_name)
    exp_fullname = ospj(exp_root, exp_dirname)
    model_path = ospj(exp_fullname, "models")
    viz_path = ospj(exp_fullname, "viz")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(viz_path, exist_ok=True)
    logger.create_log(exp_fullname)
    sys.stdout = logger
    writer = SummaryWriter(exp_fullname)

    # write cmd line
    utils.write_cmd_to_file(exp_fullname, sys.argv)

    # copy code to inside
    copyfile("./train_actor.py", ospj(model_path, "train_actor.py"))

    actor = Actor(args)
    assert args.gpus is not None
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    actor = actor.cuda()

    if args.pretrained_path is not None:
        print("Load from %s..."%(args.pretrained_path))
        actor.load_state_dict(torch.load(args.pretrained_path))
    optimizer = torch.optim.SGD(actor.parameters(), args.lr)

    train_losses = utils.AverageMeter()
    train_losses_obs = utils.AverageMeter()
    train_losses_dst = utils.AverageMeter()
    train_losses_reg = utils.AverageMeter()
    train_goal_rates = utils.AverageMeter()
    train_safe_rates = utils.AverageMeter()

    test_losses = utils.AverageMeter()
    test_losses_obs = utils.AverageMeter()
    test_losses_dst = utils.AverageMeter()
    test_losses_reg = utils.AverageMeter()
    test_goal_rates = utils.AverageMeter()
    test_safe_rates = utils.AverageMeter()

    # environment
    canvas_w=4
    canvas_h=4
    map_w = 3
    map_h = 3
    obs_x = 0
    obs_y = 0
    obs_r = args.obs_r
    dst_x = map_w / 2
    dst_y = map_h / 2
    dst_r = 0.3
    init_xmin = - map_w / 2 - map_w * args.xy_range * 0.5
    init_xmax = - map_w / 2 + map_w * args.xy_range * 0.5
    init_ymin = - map_h / 2 - map_w * args.xy_range * 0.5
    init_ymax = - map_h / 2 + map_h * args.xy_range * 0.5
    init_thmin = 0  # np.pi
    init_thmax = np.pi/2  # -np.pi
    init_vmin = 1.0
    init_vmax = 1.5

    state = random_state(args.batch_size, init_xmin, init_xmax, init_ymin, init_ymax, init_thmin, init_thmax,
                         init_vmin, init_vmax)
    state = torch.from_numpy(state).float().cuda()

    test_state = random_state(args.test_batch_size, init_xmin, init_xmax, init_ymin, init_ymax, init_thmin, init_thmax,
                              init_vmin, init_vmax)
    test_state = torch.from_numpy(test_state).float().cuda()

    # starting from a random initial condition
    for epi in range(args.num_epochs):

        state_list, u_list = collect_data(actor, state, args.t_len, args.dt, args)

        loss_obs, loss_dst, loss_reg, loss, safe_mask, goal_mask = \
            compute_losses(state_list, u_list, (obs_x, obs_y, obs_r), (dst_x, dst_y, dst_r), args)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(actor.parameters(), 1e-6)
        optimizer.step()
        optimizer.zero_grad()

        train_safe_rate, train_goal_rate = compute_metrics(safe_mask, goal_mask)
        # train_safe_rate = torch.mean(1.0 * safe_mask)
        # train_goal_rate = torch.mean(1.0 * goal_mask)

        train_losses_obs.update(loss_obs.detach().cpu().item())
        train_losses_dst.update(loss_dst.detach().cpu().item())
        train_losses_reg.update(loss_reg.detach().cpu().item())
        train_losses.update(loss.detach().cpu().item())
        train_goal_rates.update(train_goal_rate.detach().cpu().item())
        train_safe_rates.update(train_safe_rate.detach().cpu().item())

        writer.add_scalar("loss", loss, epi)
        writer.add_scalar("loss_obs", loss_obs, epi)
        writer.add_scalar("loss_dst", loss_dst, epi)
        writer.add_scalar("loss_reg", loss_reg, epi)
        writer.add_scalar("goal_rate", train_goal_rate, epi)
        writer.add_scalar("safe_rate", train_safe_rate, epi)

        # (N, T+1, K), (N, T, K)
        if epi % args.eval_freq == 0:
            actor.eval()
            test_state_list, test_u_list = collect_data(actor, test_state, args.t_len, args.dt, args)

            test_loss_obs, test_loss_dst, test_loss_reg, test_loss, test_safe_mask, test_goal_mask = \
                compute_losses(test_state_list, test_u_list, (obs_x, obs_y, obs_r), (dst_x, dst_y, dst_r), args)

            # test_safe_rate = torch.mean(1.0 * test_safe_mask)
            # test_goal_rate = torch.mean(1.0 * test_goal_mask)
            test_safe_rate, test_goal_rate = compute_metrics(test_safe_mask, test_goal_mask)

            test_losses_obs.update(test_loss_obs.detach().cpu().item())
            test_losses_dst.update(test_loss_dst.detach().cpu().item())
            test_losses_reg.update(test_loss_reg.detach().cpu().item())
            test_losses.update(test_loss.detach().cpu().item())
            test_goal_rates.update(test_goal_rate.detach().cpu().item())
            test_safe_rates.update(test_safe_rate.detach().cpu().item())

            writer.add_scalar("test_loss", test_loss, epi)
            writer.add_scalar("test_loss_obs", test_loss_obs, epi)
            writer.add_scalar("test_loss_dst", test_loss_dst, epi)
            writer.add_scalar("test_loss_reg", test_loss_reg, epi)
            writer.add_scalar("test_goal_rate", test_goal_rate, epi)
            writer.add_scalar("test_safe_rate", test_safe_rate, epi)

            torch.save(actor.state_dict(), "%s/model%d.ckpt" % (model_path, epi))
            print("%05d/%05d L %.3f(%.3f) o %.3f d %.3f r %.3f safe %.3f(%.3f) goal %.3f(%.3f) | L %.3f(%.3f) o %.3f d %.3f r %.3f safe %.3f(%.3f) goal %.3f(%.3f)"%(
                epi, args.num_epochs,
                train_losses.val, train_losses.avg, train_losses_obs.val, train_losses_dst.val,  train_losses_reg.val,
                train_safe_rates.val, train_safe_rates.avg, train_goal_rates.val, train_goal_rates.avg,
                test_losses.val, test_losses.avg, test_losses_obs.val, test_losses_dst.val, test_losses_reg.val,
                test_safe_rates.val, test_safe_rates.avg, test_goal_rates.val, test_goal_rates.avg
            ))
            writer.flush()


            # viz plot
            if epi % args.viz_freq == 0:
                for t_i in range(0, args.t_len, 1):
                    ax = plt.gcf().gca()
                    # plot goal (circ)
                    # plot obs (circ)
                    # plot init (rect)
                    circ_goal = plt.Circle((dst_x, dst_y), dst_r, color='g', alpha=0.3)
                    circ_obs = plt.Circle((obs_x, obs_y), obs_r, color='r', alpha=0.3)
                    rect_init = plt.Rectangle((init_xmin, init_ymin), init_xmax-init_xmin, init_ymax-init_ymin, color='blue', alpha=0.3)
                    ax.add_patch(circ_goal)
                    ax.add_patch(circ_obs)
                    ax.add_patch(rect_init)
                    plt.scatter(test_state_list[:,t_i,0].detach().cpu().numpy(), test_state_list[:,t_i,1].detach().cpu().numpy(), s=0.5, color="b")
                    plt.axis('scaled')
                    plt.xlim(-canvas_w/2, canvas_w/2)
                    plt.ylim(-canvas_h/2, canvas_h/2)
                    plt.savefig("%s/e%06d_t%03d.png" % (viz_path, epi, t_i), bbox_inches='tight', pad_inches=0)
                    plt.close()


                if args.show_stat:
                    state_tensor = torch.tensor(test_state_list[:,:,:4].reshape(-1,4)).cuda()
                    state_tensor.requires_grad = True
                    u_tensor = actor(state_tensor)
                    du1 = torch.autograd.grad(outputs=u_tensor[:, 0:1], inputs=state_tensor,
                                              grad_outputs=torch.ones_like(u_tensor[:, 0:1]), retain_graph=True)[0]
                    du2 = torch.autograd.grad(outputs=u_tensor[:, 1:2], inputs=state_tensor,
                                              grad_outputs=torch.ones_like(u_tensor[:, 1:2]), retain_graph=True)[0]
                    du1dth = du1.detach().cpu().numpy()[:, 2:3]
                    du2dv = du2.detach().cpu().numpy()[:, 3:4]
                    uv = u_tensor.detach().cpu().numpy()
                    print("ti %02d u-omega min|max %.4f %.4f" % (t_i, np.min(uv[:, 0]), np.max(uv[:, 0])))
                    print("ti %02d u-accel min|max %.4f %.4f" % (t_i, np.min(uv[:, 1]), np.max(uv[:, 1])))
                    print("ti %02d du1 min|max %.4f %.4f" % (t_i, np.min(du1dth[:, 0]), np.max(du1dth[:, 0])))
                    print("ti %02d du2 min|max %.4f %.4f" % (t_i, np.min(du2dv[:, 0]), np.max(du2dv[:, 0])))

    writer.close()



if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Finished in %.4f seconds"%(t2-t1))