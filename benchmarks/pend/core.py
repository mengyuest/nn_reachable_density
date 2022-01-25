# TODO Pendulum using LQR

import numpy as np
from os.path import join as ospj
from os.path import dirname as ospd

class Benchmark:
    def __init__(self, mock_args, args):
        self.n_dim = 4  # TODO(x, y, k1, k2)
        self.args = args
        self.K = np.array([[-23.58639732,  -5.31421063]])
        self.K = self.K/50

    def get_u_and_du(self, x):
        u = x[:,:2].dot(self.K.T) * np.exp(x[:,2:4])
        return u, None

    # TODO Using naive, multi-step, or odeint
    def get_next_state(self, x, u):
        g = 9.81
        L = 0.5
        m = 0.15
        # TODO(changed!) b=0.1
        b = 0.0

        dxdt = np.zeros_like(x)
        dxdt[:, 0] = x[:, 1]
        dxdt[:, 1] = (m * g * L * np.sin(x[:, 0]) - b * x[:, 1] + u[:, 0]) / (m * L ** 2)
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
        g = 9.81
        L = 0.5
        m = 0.15
        # TODO(changed!) b=0.1
        b = 0.0
        return (- b * np.ones_like(x[:, 1]) + np.exp(x[:, 3]) * self.K[0, 1]) / (m * L ** 2)

    def get_dx_and_drho(self, x_rho, t):
        state = x_rho[:-1]
        ndim = state.shape[0]
        rho = x_rho[-1]

        x = state.reshape((1, -1))

        nabla = self.get_nabla(x, None, None)
        drho = -nabla * rho

        g = 9.81
        L = 0.5
        m = 0.15
        # TODO(changed!) b=0.1
        b = 0.0

        u = x[:, :2].dot(self.K.T) * np.exp(x[:, 2:4])
        dxdt = np.zeros_like(x)
        dxdt[:, 0] = x[:, 1]
        dxdt[:, 1] = (m * g * L * np.sin(x[:, 0]) - b * x[:, 1] + u[:, 0]) / (m * L ** 2)

        dxdrho = np.zeros(ndim + 1)
        dxdrho[:ndim] = dxdt[0,:]
        dxdrho[-1] = drho
        return dxdrho  # np.concatenate((dx, drho))