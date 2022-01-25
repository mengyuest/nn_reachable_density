import numpy as np

class Benchmark:
    def __init__(self, mock_args, args):

        self.n_dim = 4  # TODO(x, y, th, error_term)
        self.args = args

        # TODO (k1, k2 changed from 1.0)
        self.k1 = 0.5
        self.k2 = 0.5
        self.k3 = 1.0
        self.v_ref = 1.0
        self.w_ref = 0

    def get_u_and_du(self, x):
        return None, None

    # TODO Using naive, multi-step, or odeint
    def get_next_state(self, x, u):
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        v_ref = self.v_ref
        w_ref = self.w_ref

        ex = x[:, 0]
        ey = x[:, 1]
        eth = x[:, 2]
        a = x[:, 3]

        d_ex = (w_ref + v_ref * (k2 * ey + k3 * np.sin(eth))) * ey - k1 * ex + a * ex  # (1-a)*(v_ref*np.cos(eth)+k1*ex)
        d_ey = -(w_ref + v_ref * (k2 * ey + k3 * np.sin(eth))) * ex + v_ref * np.sin(eth) + a * ey
        d_eth = -v_ref * (k2 * ey + k3 * np.sin(eth))
        d_a = np.zeros_like(d_eth)

        dxdt = np.stack((d_ex, d_ey, d_eth, d_a), axis=-1)

        return x + dxdt * self.args.dt

    # TODO (needed)
    def get_u_du_new_s(self, state):
        for i in range(self.args.sim_steps):
            uv, du_sum = self.get_u_and_du(state)
            new_state = self.get_next_state(state, uv)
            state = new_state
        return uv, du_sum, state

    # TODO (needed)
    def get_nabla(self, x, u, du_cache):
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        v_ref = self.v_ref
        w_ref = self.w_ref

        ex = x[:, 0]
        ey = x[:, 1]
        eth = x[:, 2]
        a = x[:, 3]

        nabla1 = - k1 + a
        nabla2 = -v_ref * k2 * ex + a
        nabla3 = -v_ref * k3 * np.cos(eth)
        return nabla1 + nabla2 + nabla3

    def get_dx_and_drho(self, x_rho, t):
        state = x_rho[:-1]
        ndim = state.shape[0]
        state=state.reshape((1, ndim))
        rho = x_rho[-1]
        nabla = self.get_nabla(state, None, None)
        drho = - nabla * rho

        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        v_ref = self.v_ref
        w_ref = self.w_ref

        ex = state[:, 0]
        ey = state[:, 1]
        eth = state[:, 2]
        a = state[:, 3]

        d_ex = (w_ref + v_ref * (k2 * ey + k3 * np.sin(eth))) * ey - k1 * ex + a * ex  # (1-a)*(v_ref*np.cos(eth)+k1*ex)
        d_ey = -(w_ref + v_ref * (k2 * ey + k3 * np.sin(eth))) * ex + v_ref * np.sin(eth) + a * ey
        d_eth = -v_ref * (k2 * ey + k3 * np.sin(eth))
        d_a = np.zeros_like(d_eth)

        dx = np.stack((d_ex, d_ey, d_eth, d_a), axis=-1)

        dxdrho = np.zeros(ndim + 1)
        dxdrho[:ndim] = dx
        dxdrho[-1] = drho
        return dxdrho  # np.concatenate((dx, drho))