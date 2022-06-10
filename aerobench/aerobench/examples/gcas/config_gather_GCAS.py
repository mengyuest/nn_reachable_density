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

# TODO(yue)
import numpy as np
import time
import test_configs.case_3m3p_extreme as cfg
# import test_configs.case_3m3p_mild as cfg
import sys
# import test_configs.case_cav as cfg

def sample_data(n, lb, ub):
    return np.random.rand(n) * (ub-lb) + lb

def function(inputs):
    # Build Initial Condition Vectors
    tmax = 3.51  # simulation time
    ap = GcasAutopilot(init_mode='roll', stdout=True, gain_str='old')
    step = 1 / 30
    init = inputs[1:]
    res = run_f16_sim(init, tmax, ap, step=step, extended_states=True)

    return inputs[0], res

class Logger(object):
    def __init__(self, path):
        self._terminal=sys.stdout
        self._log = open(path,"w")
    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)
    def flush(self):
        pass

def main():
    'main function'

    # TODO(yue) setup exp log dir
    import os
    import shutil
    from datetime import datetime
    timestr = datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S")
    log_dir="data/g%s"%(timestr)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout=Logger(os.path.join(log_dir,"simple.log"))
    shutil.copyfile('config_gather_GCAS.py', log_dir+"/config_gather_GCAS.py")
    shutil.copyfile(cfg.__file__, log_dir + "/"+cfg.__file__.split("/")[-1])

    ### Initial Conditions ###
    tt1=time.time()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=1007)
    args = parser.parse_args()

    print(args)
    print("random seed: %d"%args.random_seed)

    # np.random.seed(1007)
    np.random.seed(args.random_seed)
    # np.random.seed(1003)
    # np.random.seed(1001)

    num_trials = 5000
    num_workers = 12

    vt_sims = sample_data(num_trials, cfg.vt_min, cfg.vt_max)
    alpha_sims = sample_data(num_trials, cfg.alpha_min, cfg.alpha_max)
    beta_sims = sample_data(num_trials, cfg.beta_min, cfg.beta_max)
    phi_sims = sample_data(num_trials, cfg.phi_min, cfg.phi_max)
    theta_sims = sample_data(num_trials, cfg.theta_min, cfg.theta_max)
    psi_sims = sample_data(num_trials, cfg.psi_min, cfg.psi_max)
    p_sims = sample_data(num_trials, cfg.p_min, cfg.p_max)
    q_sims = sample_data(num_trials, cfg.q_min, cfg.q_max)
    r_sims = sample_data(num_trials, cfg.r_min, cfg.r_max)
    pn_sims = sample_data(num_trials, cfg.pn_min, cfg.pn_max)
    pe_sims = sample_data(num_trials, cfg.pe_min, cfg.pe_max)
    alt_sims = sample_data(num_trials, cfg.alt_min, cfg.alt_max)
    power_sims = sample_data(num_trials, cfg.power_min, cfg.power_max)
    nz_sims = sample_data(num_trials, cfg.nz_min, cfg.nz_max)
    ps_sims = sample_data(num_trials, cfg.ps_min, cfg.ps_max)
    ny_sims = sample_data(num_trials, cfg.ny_min, cfg.ny_max)

    res_ds = {}
    if num_workers>0:
        from multiprocessing.pool import Pool
        pool = Pool(processes=num_workers)
        inputs = [[i, vt_sims[i], alpha_sims[i], beta_sims[i], phi_sims[i], theta_sims[i],
                    psi_sims[i], p_sims[i], q_sims[i], r_sims[i], pn_sims[i], pe_sims[i],
                    alt_sims[i], power_sims[i]] for i in range(num_trials)]

        outputs = pool.map(function, inputs)

        sorted_out = sorted(outputs, key=lambda x:x[0], reverse=False)
        for i, res in sorted_out:
            for key in res:
                if key not in res_ds:
                    res_ds[key]=[]
                res_ds[key].append(res[key])
    else:
        for i in range(num_trials):
            # Build Initial Condition Vectors
            init = [vt_sims[i], alpha_sims[i], beta_sims[i], phi_sims[i], theta_sims[i],
                    psi_sims[i], p_sims[i], q_sims[i], r_sims[i], pn_sims[i], pe_sims[i],
                    alt_sims[i], power_sims[i]]
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
    t2 = time.time()

    print("alt:",np.min(res_ds["states"][:,:,11]), np.max(res_ds["states"][:,:,11]))
    print(res_ds["states"].shape)
    rate_failures = np.mean(np.min(res_ds["states"][:,:,11], axis=-1)<0)
    print("rate_failures=",rate_failures)
    np.savez("%s/sim_trajs.npz"%(log_dir), res_ds)
    tt2=time.time()
    print(f"Gathering completed in %.4f seconds (sim: %.4f   save:%.4f)"%(tt2-tt1, t2-tt1, tt2-t2))


if __name__ == '__main__':
    main()
