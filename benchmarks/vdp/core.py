# TODO Van Der Pol

import numpy as np
from os.path import join as ospj
from os.path import dirname as ospd

class Benchmark:
    def __init__(self, mock_args, args):
        self.n_dim = 2  # TODO(x, y, k1, k2)
        self.args = args
        self.mu=1.0

    def get_u_and_du(self, x):
        return None, None

    # TODO Using naive, multi-step, or odeint
    def get_next_state(self, x, u):
        dxdt = np.zeros_like(x)
        dxdt[:, 0] = x[:, 1]
        dxdt[:, 1] = self.mu * (1 - x[:, 0]**2) * x[:, 1] - x[:, 0]
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
        return self.mu * (1 - x[:, 0] ** 2)

    def get_dx_and_drho(self, x_rho, t):
        state = x_rho[:-1]
        ndim = state.shape[0]
        x = state.reshape((1, ndim))
        rho = x_rho[-1]
        nabla = self.get_nabla(x, None, None)
        drho = - nabla * rho

        dxdt = np.zeros_like(x)
        dxdt[:, 0] = x[:, 1]
        dxdt[:, 1] = self.mu * (1 - x[:, 0] ** 2) * x[:, 1] - x[:, 0]

        dxdrho = np.zeros(ndim + 1)
        dxdrho[:ndim] = dxdt[0,:]
        dxdrho[-1] = drho
        return dxdrho  # np.concatenate((dx, drho))