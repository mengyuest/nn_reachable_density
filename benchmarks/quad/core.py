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
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(6, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Benchmark:
    def __init__(self, mock_args, args):
        self.GRAVITY = 9.8

        local_dir=ospd(__file__)
        actor = Actor()
        file_h5 = h5py.File(ospj(local_dir, "model.h5"), "r")
        actor.state_dict()["fc1.weight"][:] = torch.from_numpy(np.array(file_h5['dense']['dense']['kernel:0'])).T
        actor.state_dict()["fc1.bias"][:] = torch.from_numpy(np.array(file_h5['dense']['dense']['bias:0']))
        actor.state_dict()["fc2.weight"][:] = torch.from_numpy(np.array(file_h5['dense_1']['dense_1']['kernel:0'])).T
        actor.state_dict()["fc2.bias"][:] = torch.from_numpy(np.array(file_h5['dense_1']['dense_1']['bias:0']))
        actor.state_dict()["fc3.weight"][:] = torch.from_numpy(np.array(file_h5['dense_2']['dense_2']['kernel:0'])).T
        actor.state_dict()["fc3.bias"][:] = torch.from_numpy(np.array(file_h5['dense_2']['dense_2']['bias:0']))

        actor.eval()

        self.n_dim = 6  # TODO(x, y, z, vx, vy, vz)
        self.actor = actor
        self.args = args


    def get_u_and_du(self, x):
        x_tensor = torch.from_numpy(x).float()
        x_tensor.requires_grad = True
        u_tensor = self.actor(x_tensor)
        dudx_list=[]
        for i in range(3):
            du = torch.autograd.grad(outputs=u_tensor[:, i:i+1], inputs=x_tensor,
                                      grad_outputs=torch.ones_like(u_tensor[:, i:i+1]), retain_graph=True)[0]
            dudx = du.detach().numpy()[:, i+3]
            dudx_list.append(dudx)
        uv = u_tensor.detach().numpy()

        return uv, dudx_list

    # TODO Using naive, multi-step, or odeint
    def get_next_state(self, x, u):
        dpx = x[:, 3]
        dpy = x[:, 4]
        dpz = x[:, 5]
        dvx = self.GRAVITY * u[:, 0]
        dvy = - self.GRAVITY * u[:, 1]
        dvz = u[:, 2] - self.GRAVITY

        new_x = np.array(x)
        new_x[:, 0] += dpx * self.args.dt
        new_x[:, 1] += dpy * self.args.dt
        new_x[:, 2] += dpz * self.args.dt
        new_x[:, 3] += dvx * self.args.dt
        new_x[:, 4] += dvy * self.args.dt
        new_x[:, 5] += dvz * self.args.dt

        return new_x

    # TODO (needed)
    def get_u_du_new_s(self, x):
        for _ in range(self.args.sim_steps):
            u, du_cache = self.get_u_and_du(x)
            new_x = self.get_next_state(x, u)
            x = new_x
        return u, du_cache, x


    # TODO (needed)
    def get_nabla(self, x, u, du_cache):
        du1_dvx, du2_dvy, du3_dvz = du_cache
        nabla = du1_dvx * self.GRAVITY - du2_dvy * self.GRAVITY + du3_dvz
        return nabla

    def get_dx_and_drho(self, x_rho, t):
        state = x_rho[:-1]
        ndim = state.shape[0]
        rho = x_rho[-1]

        x = state.reshape((1, -1))

        x_tensor = torch.from_numpy(x).float()
        x_tensor.requires_grad = True
        u_tensor = self.actor(x_tensor)
        dudx_list = []
        for i in range(3):
            du = torch.autograd.grad(outputs=u_tensor[:, i:i + 1], inputs=x_tensor,
                                     grad_outputs=torch.ones_like(u_tensor[:, i:i + 1]), retain_graph=True)[0]
            dudx = du.detach().numpy()[:, i + 3]
            dudx_list.append(dudx)
        uv = u_tensor.detach().numpy()

        nabla = self.get_nabla(x, uv, dudx_list)

        drho = -nabla * rho

        dxdrho = np.zeros(ndim + 1)
        dxdrho[0] = x[0, 3]
        dxdrho[1] = x[0, 4]
        dxdrho[2] = x[0, 5]
        dxdrho[3] = self.GRAVITY * uv[0, 0]
        dxdrho[4] = - self.GRAVITY * uv[0, 1]
        dxdrho[5] = uv[0, 2] - self.GRAVITY
        dxdrho[6] = drho
        return dxdrho  # np.concatenate((dx, drho))
