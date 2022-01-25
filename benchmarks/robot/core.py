import numpy as np
import torch
from torch import nn


# TODO (temporary solution)
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

class Benchmark:
    def __init__(self, mock_args, args):
        mock_args.hiddens = [32, 32]
        mock_args.output_type = "tanh"
        mock_args.tanh_gain = 4.0
        actor = Actor(mock_args)
        actor.load_state_dict(torch.load(args.pretrained_path))
        actor.eval()

        self.n_dim = 4  # TODO(x, y, th, v)
        self.actor = actor
        self.args = args

    def get_u_and_du(self, state):
        state_tensor = torch.from_numpy(state).float()
        state_tensor.requires_grad = True
        u_tensor = self.actor(state_tensor)
        du1 = torch.autograd.grad(outputs=u_tensor[:, 0:1], inputs=state_tensor,
                                  grad_outputs=torch.ones_like(u_tensor[:, 0:1]), retain_graph=True)[0]
        du2 = torch.autograd.grad(outputs=u_tensor[:, 1:2], inputs=state_tensor,
                                  grad_outputs=torch.ones_like(u_tensor[:, 1:2]), retain_graph=True)[0]

        du1dth = du1.detach().numpy()[:, 2:3]
        du2dv = du2.detach().numpy()[:, 3:4]
        uv = u_tensor.detach().numpy()
        return uv, [du1dth, du2dv]

    # TODO Using naive, multi-step, or odeint
    def get_next_state(self, state, uv):
        dx = state[:, 3] * np.cos(state[:, 2])
        dy = state[:, 3] * np.sin(state[:, 2])
        dth = uv[:, 0]
        dv = uv[:, 1]
        new_state = np.array(state)
        new_state[:, 0] += dx * self.args.dt
        new_state[:, 1] += dy * self.args.dt
        new_state[:, 2] += dth * self.args.dt
        new_state[:, 3] += dv * self.args.dt

        return new_state

    # TODO (needed)
    def get_u_du_new_s(self, state):
        for i in range(self.args.sim_steps):
            uv, du_sum = self.get_u_and_du(state)
            new_state = self.get_next_state(state, uv)
            state = new_state
        return uv, du_sum, state

    # TODO (needed)
    def get_nabla(self, state, u, du_cache):
        du1dth, du2dv = du_cache
        return (du1dth + du2dv).flatten()

    # TODO(dbg for ode)
    def get_dx_and_drho(self, x_rho, t):
        state = x_rho[:-1]
        ndim = state.shape[0]
        rho = x_rho[-1]
        state_tensor = torch.from_numpy(state.reshape((1, ndim))).float()
        state_tensor.requires_grad = True
        u_tensor = self.actor(state_tensor)
        du1 = torch.autograd.grad(outputs=u_tensor[:, 0:1], inputs=state_tensor,
                                  grad_outputs=torch.ones_like(u_tensor[:, 0:1]), retain_graph=True)[0]
        du2 = torch.autograd.grad(outputs=u_tensor[:, 1:2], inputs=state_tensor,
                                  grad_outputs=torch.ones_like(u_tensor[:, 1:2]), retain_graph=True)[0]
        du1dth = du1.detach().numpy()[0, 2:3]
        du2dv = du2.detach().numpy()[0, 3:4]
        nabla = (du1dth + du2dv)
        drho = - nabla * rho

        # dx = state[:, 3] * np.cos(state[:, 2])
        # dy = state[:, 3] * np.sin(state[:, 2])
        # dth = uv[:, 0]
        # dv = uv[:, 1]
        #
        u = u_tensor.detach().cpu().numpy().flatten()
        dx = np.zeros((ndim,))
        dx[0] = state[3] * np.cos(state[2])
        dx[1] = state[3] * np.sin(state[2])
        dx[2] = u[0]
        dx[3] = u[1]

        dxdrho = np.zeros(ndim+1)
        dxdrho[:ndim] = dx
        dxdrho[-1] = drho
        return dxdrho  #np.concatenate((dx, drho))

    # TODO