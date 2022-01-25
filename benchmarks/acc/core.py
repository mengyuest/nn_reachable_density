import h5py
import numpy as np
import torch
from torch import nn
from os.path import join as ospj
from os.path import dirname as ospd
import scipy.io


# TODO (temporary solution)
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(5, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 20)
        self.fc5 = nn.Linear(20, 20)
        self.fc6 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)

        return x

class Benchmark:
    def __init__(self, mock_args, args):
        local_dir=ospd(__file__)
        actor = Actor()
        num_layers=5
        nn_mat = scipy.io.loadmat("%s/controller_%d_20.mat" % (local_dir, num_layers))
        for i in range(num_layers + 1):
            actor.state_dict()["fc%s.weight" % (i + 1)][:] = torch.from_numpy(
                np.array(nn_mat['network']['weights'][0, 0][0, i]))
            actor.state_dict()["fc%s.bias" % (i + 1)][:] = torch.from_numpy(
                np.array(nn_mat['network']['bias'][0, 0][i, 0][:, 0]))

        actor.eval()

        self.n_dim = 7  # TODO(x, y, th, v)
        self.actor = actor
        self.args = args


    def get_u_and_du(self, x):
        x_tensor = torch.from_numpy(x).float()
        x_rel, v_lead, r_lead, v_ego, r_ego, a_lead, tau = torch.split(x_tensor, [1, 1, 1, 1, 1, 1, 1], dim=-1)
        r_ego.requires_grad = True
        v_ref = torch.ones_like(v_lead) * 30.0
        t_ref = torch.ones_like(v_lead) * 1.4
        tilde_x_rel = x_rel  # - r_ego * tau**2
        tilde_v_rel = v_lead - v_ego - r_ego * tau
        tilde_v_ego = v_ego + r_ego * tau

        u_input = torch.cat([v_ref, t_ref, tilde_x_rel, tilde_v_rel, tilde_v_ego], axis=-1)
        u_tensor = self.actor(u_input)

        dudr_ego = torch.autograd.grad(outputs=u_tensor[:, 0:1], inputs=r_ego,
                                       grad_outputs=torch.ones_like(u_tensor[:, 0:1]), retain_graph=True)[0]
        dudr_ego = dudr_ego.detach().cpu().numpy()

        u = u_tensor.detach().cpu().numpy()

        return u, [dudr_ego]

    # TODO Using naive, multi-step, or odeint
    def get_next_state(self, x, u):
        d_rel, v_lead, r_lead, v_ego, r_ego, a_lead, tau = np.split(x, 7, axis=-1)
        d_d_rel = v_lead - v_ego
        d_v_lead = r_lead
        d_r_lead = a_lead
        d_v_ego = r_ego
        d_r_ego = -2 * r_ego + 2 * u
        d_a = -2 * r_lead
        d_tau = 0.0 * np.zeros_like(tau)
        dxdt = np.concatenate((d_d_rel, d_v_lead, d_r_lead, d_v_ego, d_r_ego, d_a, d_tau), axis=-1)
        new_x = np.array(x)
        return new_x + dxdt * self.args.dt



    # TODO (needed)
    def get_u_du_new_s(self, x):
        for _ in range(self.args.sim_steps):
            u, du_cache = self.get_u_and_du(x)
            new_x = self.get_next_state(x, u)
            x = new_x
        return u, du_cache, x

    # TODO (needed)
    def get_nabla(self, x, u, du_cache):
        nabla = np.ones_like(x[:, 0:1]) * -2 + 2 * du_cache[0]
        return nabla[:,0]

    def get_dx_and_drho(self, x_rho, t):
        state = x_rho[:-1]
        ndim = state.shape[0]
        rho = x_rho[-1]

        x = state.reshape((1, -1))

        u, du_cache = self.get_u_and_du(x)
        nabla = self.get_nabla(x, u, du_cache)
        drho = -nabla * rho



        d_rel, v_lead, r_lead, v_ego, r_ego, a_lead, tau = np.split(x, 7, axis=-1)
        d_d_rel = v_lead - v_ego
        d_v_lead = r_lead
        d_r_lead = a_lead
        d_v_ego = r_ego
        d_r_ego = -2 * r_ego + 2 * u
        d_a = -2 * r_lead
        d_tau = 0.0 * np.zeros_like(tau)
        dxdt = np.concatenate((d_d_rel, d_v_lead, d_r_lead, d_v_ego, d_r_ego, d_a, d_tau), axis=-1)


        dxdrho = np.zeros(ndim + 1)
        dxdrho[:ndim] = dxdt[0, :]
        dxdrho[-1] = drho
        return dxdrho  # np.concatenate((dx, drho))
