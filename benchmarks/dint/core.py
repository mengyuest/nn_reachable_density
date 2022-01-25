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
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.maximum(torch.minimum(x, torch.ones_like(x)), -1.0*torch.ones_like(x))

class Benchmark:
    def __init__(self, mock_args, args):
        local_dir=ospd(__file__)
        actor = Actor()
        file_h5 = h5py.File(ospj(local_dir, "model.h5"), "r")
        actor.state_dict()["fc1.weight"][:] = torch.from_numpy(np.array(file_h5['dense_1']['dense_1_1']['kernel:0'])).T
        actor.state_dict()["fc1.bias"][:] = torch.from_numpy(np.array(file_h5['dense_1']['dense_1_1']['bias:0']))
        actor.state_dict()["fc2.weight"][:] = torch.from_numpy(np.array(file_h5['dense_2']['dense_2_1']['kernel:0'])).T
        actor.state_dict()["fc2.bias"][:] = torch.from_numpy(np.array(file_h5['dense_2']['dense_2_1']['bias:0']))
        actor.state_dict()["fc3.weight"][:] = torch.from_numpy(np.array(file_h5['dense_3']['dense_3_1']['kernel:0'])).T
        actor.state_dict()["fc3.bias"][:] = torch.from_numpy(np.array(file_h5['dense_3']['dense_3_1']['bias:0']))
        actor.eval()

        self.n_dim = 2  # TODO(x, y)
        self.actor = actor
        self.args = args

        self.A = np.array([[0.0, 1.0], [0.0, 0.0]])
        self.B = np.array([[0.5], [1.0]])


    def get_u_and_du(self, x):
        x_tensor = torch.from_numpy(x).float()
        x_tensor.requires_grad = True
        u_tensor = self.actor(x_tensor)
        dudx_tensor = torch.autograd.grad(outputs=u_tensor[:, 0:1], inputs=x_tensor,
                                   grad_outputs=torch.ones_like(u_tensor[:, 0:1]), retain_graph=True)[0]
        u = u_tensor.detach().cpu().numpy()
        dudx = dudx_tensor.detach().cpu().numpy()
        return u, [dudx]

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
        return 0.5 * du_cache[0][:, 0] + 1.0 * du_cache[0][:, 1]

    def get_dx_and_drho(self, x_rho, t):
        state = x_rho[:-1]
        ndim = state.shape[0]
        rho = x_rho[-1]

        x_tensor = torch.from_numpy(state.reshape((1, ndim))).float()
        x_tensor.requires_grad = True
        u_tensor = self.actor(x_tensor)
        dudx_tensor = torch.autograd.grad(outputs=u_tensor[:, 0:1], inputs=x_tensor,
                                          grad_outputs=torch.ones_like(u_tensor[:, 0:1]), retain_graph=True)[0]
        u = u_tensor.detach().cpu().numpy()
        dudx = dudx_tensor.detach().cpu().numpy()
        nabla = 0.5 * dudx[:, 0] + 1.0 * dudx[:, 1]

        drho = -nabla * rho

        dx = np.dot(state, self.A.T) + np.dot(u, self.B.T)

        dxdrho = np.zeros(ndim + 1)
        dxdrho[:ndim] = dx
        dxdrho[-1] = drho
        return dxdrho  # np.concatenate((dx, drho))
