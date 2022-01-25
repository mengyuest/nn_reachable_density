import h5py
import numpy as np
import torch
from torch import nn
from os.path import join as ospj
from os.path import dirname as ospd

# TODO (temporary solution)
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(15, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.fc3 = nn.Linear(300, 200)
        self.bn3 = nn.BatchNorm1d(200)
        self.fc4 = nn.Linear(200, 8)

        self.action_bound = 10.0

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.bn1(x)

        # TODO normalization
        x = self.relu(self.fc2(x))
        x = self.bn2(x)

        # TODO normalization
        x = self.relu(self.fc3(x))
        x = self.bn3(x)

        # TODO normalization
        x = self.tanh(self.fc4(x)) * self.action_bound

        return x

class Benchmark:
    def __init__(self, mock_args, args):
        local_dir=ospd(__file__)
        actor = Actor()
        ckpt = np.load(ospj(local_dir, "final_model8.npz"), allow_pickle=True)
        ckpt = ckpt['arr_0'].item()
        actor.eval()

        str1 = ["1", "2", "3", "4"]
        str2 = ["", "_1", "_2", "_3"]
        for i in range(4):
            actor.state_dict()["fc%s.weight" % (str1[i])][:] = torch.from_numpy(
                ckpt["FullyConnected%s/W" % (str2[i])].T)
            actor.state_dict()["fc%s.bias" % (str1[i])][:] = torch.from_numpy(ckpt["FullyConnected%s/b" % (str2[i])])
            if i < 3:
                actor.state_dict()["bn%s.weight" % (str1[i])][:] = torch.from_numpy(
                    ckpt["BatchNormalization%s/gamma" % (str2[i])])
                actor.state_dict()["bn%s.bias" % (str1[i])][:] = torch.from_numpy(
                    ckpt["BatchNormalization%s/beta" % (str2[i])])
                actor.state_dict()["bn%s.running_mean" % (str1[i])][:] = torch.from_numpy(
                    ckpt["BatchNormalization%s/moving_mean" % (str2[i])])
                actor.state_dict()["bn%s.running_var" % (str1[i])][:] = torch.from_numpy(
                    ckpt["BatchNormalization%s/moving_variance" % (str2[i])])

        self.n_dim = 15 + 1  # TODO(v, dx, dv, dx, dv, ...) + param
        self.output_dim = 8
        self.actor = actor
        self.args = args

        self.A = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

        self.B = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.pairs = []
        for row_i, row in enumerate(self.B):
            for col_j, col in enumerate(row):
                if col == 1:
                    self.pairs.append((1, col_j, row_i))
                elif col == -1:
                    self.pairs.append((-1, col_j, row_i))


    def get_u_and_du(self, x):
        x_tensor = torch.from_numpy(x[:,:self.n_dim-1]).float()
        x_tensor.requires_grad = True
        u_tensor = self.actor(x_tensor)
        u = u_tensor.detach().cpu().numpy()
        du_list = [torch.autograd.grad(outputs=u_tensor[:, k:k + 1], inputs=x_tensor,
                                       grad_outputs=torch.ones_like(u_tensor[:, k:k + 1]), retain_graph=True)[
                       0].detach().numpy()
                   for k in range(self.output_dim)]
        dudx_list = [pa[0] * du_list[pa[1]][:, pa[2]:pa[2] + 1] for pa in self.pairs]
        return u, dudx_list

    # TODO Using naive, multi-step, or odeint
    def get_next_state(self, x, u):
        dxdt = np.dot(x, self.A.T) + np.dot(u, self.B.T)
        return x + dxdt * self.args.dt

    # TODO (needed)
    def get_u_du_new_s(self, x):
        for _ in range(self.args.sim_steps):
            u, du_cache = self.get_u_and_du(x)
            new_x = self.get_next_state(x, u)
            x = new_x
        return u, du_cache, x

    # TODO (needed)
    def get_nabla(self, x, u, du_cache):
        return np.sum(du_cache, axis=0).flatten()

    def get_dx_and_drho(self, x_rho, t):
        state = x_rho[:-1]
        ndim = state.shape[0]
        rho = x_rho[-1]
        x = state.reshape((1, ndim))
        u, dudx_list = self.get_u_and_du(x)
        nabla = self.get_nabla(x, u, dudx_list)

        drho = -nabla * rho

        dx = np.dot(x, self.A.T) + np.dot(u, self.B.T)

        dxdrho = np.zeros(ndim + 1)
        dxdrho[:ndim] = dx[0,:]
        dxdrho[-1] = drho
        return dxdrho  # np.concatenate((dx, drho))
