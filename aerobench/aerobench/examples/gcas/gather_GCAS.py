'''
Stanley Bak

should match 'GCAS' scenario from matlab version
'''

import math

from numpy import deg2rad
import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim

from aerobench.visualize import plot

from gcas_autopilot import GcasAutopilot

def main():
    'main function'

    ### Initial Conditions ###
    power = 9 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 1000        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = -math.pi/8           # Roll angle from wings level (rad)
    theta = (-math.pi/2)*0.3         # Pitch angle from nose level (rad)
    psi = 0   # Yaw angle from North (rad)

    # TODO(yue)
    import numpy as np
    import time
    tt1=time.time()
    np.random.seed(1007)
    num_trials = 1000

    power_sim = power
    alpha_min = alpha * 0.9
    alpha_max = alpha * 1.1
    alt_min = alt * 0.9
    alt_max = alt * 1.1
    vt_min = vt * 0.9
    vt_max = vt * 1.1
    phi_min = phi * 1.1
    phi_max = phi * 0.9
    theta_min = theta * 1.1
    theta_max = theta * 0.9
    psi_sim = psi

    beta_sim = beta

    alpha_sims = np.random.rand(num_trials) * (alpha_max-alpha_min) + alpha_min
    alt_sims = np.random.rand(num_trials) * (alt_max - alt_min) + alt_min
    vt_sims = np.random.rand(num_trials) * (vt_max - vt_min) + vt_min
    phi_sims = np.random.rand(num_trials) * (phi_max - phi_min) + phi_min
    theta_sims = np.random.rand(num_trials) * (theta_max - theta_min) + theta_min

    res_ds={}
    for i in range(num_trials):
        # Build Initial Condition Vectors
        # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
        init = [vt_sims[i], alpha_sims[i], beta_sim,
                phi_sims[i], theta_sims[i], psi_sim, 0, 0, 0, 0, 0, alt_sims[i], power_sim]
        tmax = 3.51  # simulation time

        ap = GcasAutopilot(init_mode='roll', stdout=True, gain_str='old')

        step = 1/30
        res = run_f16_sim(init, tmax, ap, step=step, extended_states=True)
        for key in res:
            if key not in res_ds:
                res_ds[key]=[]
            res_ds[key].append(res[key])

    for key in res_ds:
        res_ds[key] = np.stack(res_ds[key], axis=0)

    # import matplotlib.pyplot as plt
    # ymins=[-np.min(res_ds["grad_list"].reshape((num_trials*106, 16, 16))[:,i,i])+np.mean(res_ds["grad_list"].reshape((num_trials*106, 16, 16))[:,i,i]) for i in range(16)]
    # ymaxs=[np.max(res_ds["grad_list"].reshape((num_trials*106, 16, 16))[:,i,i])-np.mean(res_ds["grad_list"].reshape((num_trials*106, 16, 16))[:,i,i]) for i in range(16)]
    # plt.bar(range(16), [np.mean(res_ds["grad_list"].reshape((num_trials*106, 16, 16))[:,i,i]) for i in range(16)], yerr=[ymins, ymaxs], capsize=5)
    # # plt.bar(range(16), [np.max(res_ds["grad_list"].reshape((num_trials * 106, 16, 16))[:, i, i]) for i in range(16)], color="red")
    # plt.xlabel("index")
    # plt.ylabel("gradient value")
    # plt.show()

    np.savez("sim_trajs_f_N%d.npz"%(num_trials), res_ds)
    tt2=time.time()
    print(f"Gathering completed in %.4f seconds"%(tt2-tt1))

    # plot.plot_single(res, 'alt', title='Altitude (ft)')
    # filename = 'alt.png'
    # plt.savefig(filename)
    # print(f"Made {filename}")
    #
    # plot.plot_attitude(res)
    # filename = 'attitude.png'
    # plt.savefig(filename)
    # print(f"Made {filename}")
    #
    # # plot inner loop controls + references
    # plot.plot_inner_loop(res)
    # filename = 'inner_loop.png'
    # plt.savefig(filename)
    # print(f"Made {filename}")
    #
    # # plot outer loop controls + references
    # plot.plot_outer_loop(res)
    # filename = 'outer_loop.png'
    # plt.savefig(filename)
    # print(f"Made {filename}")

if __name__ == '__main__':
    main()
